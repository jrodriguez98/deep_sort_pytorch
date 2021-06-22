import numpy as np

COLORS = np.random.uniform(0, 255, size=(50, 3))

VECTORES_INTERES = [
        '4x4',
        'camion',
        'carro_de_combate_oruga',
        'tanqueta_oruga',
        'tanqueta_ruedas',
        'vehiculo_civil',
        'moto'
    ]

CATEGORIES = [
    {'id': 1, 'name': '4x4'},
    {'id': 2, 'name': 'camion'},
    {'id': 3, 'name': 'carro_de_combate_oruga'},
    {'id': 4, 'name': 'tanqueta_oruga'},
    {'id': 5, 'name': 'tanqueta_ruedas'},
    {'id': 6, 'name': 'vehiculo_civil'},
    {'id': 7, 'name': 'moto'},
]


def get_deep_format(objects_detected):
    """ Función para transformar las detecciones al formato del DeepSort.

                Parameters
                ----------
                objects_detected: list
                    Objectos detectados en un frame

                Returns
                -------
                bbox_xywh: np.array
                    Bboxes siendo xy el centro del recuadro y wh la anchura y altura del recuadro
                cls_conf: np.array
                    Confianza de las detecciones de los objetos detectados
                cls_ids: np.array
                    Identificadores de los objetos detectados
        """

    if len(objects_detected) == 0:
        deep_object_format, cls_conf, cls_ids = np.array([]), np.array([]), None
    else:
        deep_object_format = objects_detected[:, 0:4]  # xyxy
        deep_object_format[:, 2:] -= deep_object_format[:, 0:2]  # blwh
        deep_object_format[:, :2] += np.asarray(deep_object_format[:, 2:]/2, dtype=int)  #xywh

        deep_object_format[:, 2:] *= 1.1

        cls_conf = objects_detected[:, 4]
        cls_ids = objects_detected[:, 5]

    return deep_object_format, cls_conf, cls_ids
