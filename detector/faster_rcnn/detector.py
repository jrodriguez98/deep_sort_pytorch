import numpy as np
from PIL import Image
import time
import tensorflow as tf
import os
import cv2


def load_model_in_gpu(IAModel):
    """Función para cargar el modelo de IA en la GPU

        Parameters
        ----------
        IAModel: str
            Ruta al directorio del modelo

        Returns
        -------
        function
            Función de detección del modelo
    """

    # Cargamos el fichero de configuración
    pipeline_config = os.path.join(IAModel, 'model.config')

    # Reservamos espacio en la gpu para el modelo
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
        except RuntimeError as e:
            print(e)

    # Cargamos archivos de configuración del modelo
    from object_detection.utils import config_util
    from object_detection.builders import model_builder

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restauramos el ckpt
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(IAModel, 'ckpt-0')).expect_partial()

    def get_model_detection_function(model):
        """Get a tf.function for detection."""

        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            detections = model.postprocess(prediction_dict, shapes)

            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn

    detect_fn = get_model_detection_function(detection_model)

    return detect_fn


def tensorflow_detection_tf2(value_threshold, dict_images, num_classes, vectores_interes,
                             categories, max_width_crop, max_height_crop, detect_fn):
    """Función para realizar la detección en imágenes con tensorflow.

        Parameters
        ----------
        value_threshold: str
            Valor umbral para detectar un objeto entre 0 y 1
        dict_images: dict
            Diccionario con las imágenes y sus códigos
        num_classes: int
            Número de clases posibles a ser detectadas
        vectores_interes: list
            Lista con los vectores de interés que serán retornados en la detección
        categories: list
            Clases posibles a ser detectadas
        max_width_crop: int
            Valor máximo de ancho para realizar el crop, mayor que 0
        max_height_crop: int
            Valor máximo de alto para realizar el crop, mayor que 0
        detect_fn: function
            Función de detección del modelo

        Returns
        -------
        list
            Lista de listas con vectores detectados por cada imagen en la que se ha realizado detección

    """

    # Abrimos el grafo de detección
    detection_result = []
    value_threshold = float(value_threshold)

    for contador_imagen, image in dict_images.items():
        # En obtenemos los atributos de la imagen
        if type(dict_images) == dict:  # Imágenes
            image_code = str(contador_imagen)
            image = dict_images[str(contador_imagen)]
            (width, height) = image.size
            #print("DEBUG: detectando imagen " + str(contador_imagen) + '/' + str(len(dict_images.keys())))

        # Preparamos las variables a usar
        num_rows = int(width / (max_width_crop / 2))
        num_cols = int(height / (max_height_crop / 2))
        if num_cols == 0:
            num_cols = 1
        if num_rows == 0:
            num_rows = 1
        crop_x = 0
        crop_y = 0
        objects_detected = []
        objects_detected_dict = {}
        clases_detected = []

        # Realizamos la detección por crop
        for row in range(num_rows):
            # Ajustamos las imágenes de crop para que tengan las mismas dimensiones
            last_crop_x = crop_x
            for col in range(num_cols):
                #print("DEBUG: detectando crop row-col", row + 1, col + 1, "de", num_rows, num_cols)
                # Ajustamos hasta donde se va a cortar la imagen
                top_crop_x = crop_x + max_width_crop
                top_crop_y = crop_y + max_height_crop
                # Nos aseguramos de que no se vaya a recortar más de las dimensiones de las imágenes
                if top_crop_x > width:
                    top_crop_x = width
                if top_crop_y > height:
                    top_crop_y = height
                # Ajustamos las imágenes de crop para que tengan las mismas dimensiones
                last_crop_y = crop_y
                if max_width_crop > top_crop_x - crop_x:
                    crop_x = top_crop_x - max_width_crop
                if max_height_crop > top_crop_y - crop_y:
                    crop_y = top_crop_y - max_height_crop
                # Obtenemos la imagen cropeada y la detectamos
                crop_image = image.crop((crop_x, crop_y, top_crop_x, top_crop_y))
                (crop_width, crop_height) = crop_image.size
                # image_np = np.array(crop_image.getdata()).reshape((crop_height, crop_width, 3)).astype(np.uint8)
                image_np = np.array(crop_image).astype(np.uint8)

                input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                init_time_det = time.time()
                detections, predictions_dict, shapes = detect_fn(input_tensor)
                #print("DEBUG: Tiempo solo de deteccion:", time.time() - init_time_det)
                label_id_offset = 1

                # Una vez detectado el crop, sacamos los objetos que se han detectado para guardarlos
                x_height, x_width, channels = image_np.shape
                for i in range((np.squeeze(detections['detection_boxes'][0].numpy())).shape[0]):
                    if (np.squeeze(detections['detection_scores'][0].numpy())[i]) > value_threshold:
                        nombre_clase = np.squeeze((detections['detection_classes'][0].numpy() +
                                                   label_id_offset).astype(int)).astype(np.int32)[i] - 1
                        if nombre_clase <= num_classes:
                            for elem_list in categories:
                                if elem_list['id'] == (nombre_clase + 1):
                                    name_class = elem_list['name']
                            class_id = (nombre_clase + 1)
                            # Incluir solo los vectores de interés en la predicción
                            if name_class in vectores_interes:
                                x_min = int(np.squeeze(detections['detection_boxes'][0].numpy())[i][1] * x_width)
                                x_max = int(np.squeeze(detections['detection_boxes'][0].numpy())[i][3] * x_width)
                                y_min = int(np.squeeze(detections['detection_boxes'][0].numpy())[i][0] * x_height)
                                y_max = int(np.squeeze(detections['detection_boxes'][0].numpy())[i][2] * x_height)
                                final_string = str(x_min + crop_x) + '_' + str(y_min + crop_y) + '_' + \
                                               str(x_max + crop_x) + '_' + str(y_max + crop_y) + '_' + \
                                               str(np.squeeze(detections['detection_scores'][0].numpy())[i]) + '_' + \
                                               str(class_id)

                                if name_class in objects_detected_dict:
                                    if final_string in objects_detected_dict[name_class]:
                                        pass
                                    else:
                                        aux_list = objects_detected_dict[name_class]
                                        aux_list.append(final_string)
                                        objects_detected_dict[name_class] = aux_list
                                else:
                                    objects_detected_dict[name_class] = [final_string]
                                clases_detected.append(np.squeeze(
                                    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)).astype(
                                    np.int32)[i])
                            else:
                                print("DEBUG: Detectada clase de no interes:", name_class)
                # Actualizamos el siguiente crop a realizar para la misma fila
                crop_y = last_crop_y + int(max_height_crop / 2)
            # Actualizamos el siguiente crop a realizar avanzando de fila
            crop_x = last_crop_x + int(max_width_crop / 2)
            crop_y = 0
        # Montamos los datos para la imagen final con los bounding box
        final_box = ""
        final_scores = ""
        final_clases = ""
        count_objet = 0
        for name_class in objects_detected_dict:
            for obj in objects_detected_dict[name_class]:
                split_obj = obj.split('_')
                nuevo_objeto = [name_class, int(split_obj[0]), int(split_obj[1]), int(split_obj[2]), int(split_obj[3]),
                                split_obj[4], split_obj[5]]
                objects_detected.append(nuevo_objeto)
        for object in objects_detected:
            aux_box = np.array([[object[2] / height, object[1] / width, object[4] / height, object[3] / width]])
            aux_scores = np.array([float(object[5])])
            aux_classes = np.array([clases_detected[count_objet]])
            if type(final_box) == type(aux_box):
                final_box = np.append(final_box, aux_box, axis=0)
                final_scores = np.append(final_scores, aux_scores, axis=0)
                final_clases = np.append(final_clases, aux_classes, axis=0)
            else:
                final_box = aux_box
                final_scores = aux_scores
                final_clases = aux_classes
            count_objet += 1


        # Preparamos los metadatos para devolverlos
        detection_result = {
            "Id_imagen": image_code,
            "objects_detected": [],
            "class_names": []
        }

        for object_detected in objects_detected:
            detection_result["objects_detected"].append([int(object_detected[1]), int(object_detected[2]),
                              int(object_detected[3]), int(object_detected[4]), float(object_detected[5]), int(object_detected[6])])
            detection_result["class_names"].append(object_detected[0])

        detection_result["objects_detected"] = np.array(detection_result["objects_detected"])

    return detection_result
