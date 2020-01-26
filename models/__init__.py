from .efficientdet import EfficientDet
from .efficientdet import EFFICIENTDET
from .matroid import MatroidModel


def get_model(args):
    model = EfficientDet(num_classes=args.num_class,
                         network=args.model,
                         W_bifpn=EFFICIENTDET[args.model]['W_bifpn'],
                         D_bifpn=EFFICIENTDET[args.model]['D_bifpn'],
                         D_class=EFFICIENTDET[args.model]['D_class']
                         )
    return model

