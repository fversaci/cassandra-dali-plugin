import torchvision.models as models
import torch.nn as nn
"""
This class is used to remove the classifier layer(s) from models
"""
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
"""
This helper function sets the .requires_grad attribute of the parameters
in the model to False when we are freeze_params. 
By default, when we load a pretrained model all of the parameters
have .requires_grad=True, which is fine if we are training from scratch or finetuning.
However, if we are freeze_params and only want to compute gradients for the newly
initialized layer then we want all of the other parameters to not require gradients. 
This will make more sense later.
"""
def set_parameter_requires_grad(model, freeze_params):
    # if feature_extractins is True all model parameters are frozen
    if freeze_params:
        for param in model.parameters():
            param.requires_grad = False
            

def initialize_model(model_name, num_classes, freeze_params=False, pretrained_weights=None, new_top=True, weights_fn=None, get_features=False):
    """
    A model from pretrained torchvision models is instatiated. It is possibile to 
    freeze all parameters but the classifiers one as well as load a specific set of weights.
    
    @ model_name: one of the torchvision models among vggs, resnets, inceptions and efficientnets
    @ num_classes: number of classes of the output classifier
    @ freeze_params: if True model parameters are set as not trainable
    @ weights: the name of the weights to be used (None to start without pretrained data)
    @ new_top: if True a new classifier part is created with num_classes as output size and with trainable params
    
    returns the model object
    """
    
    model_ft = None

    if "resnet" in model_name:
        if model_name == 'resnet18':
            model_ft = models.resnet18(weights=pretrained_weights)
        elif model_name == 'resnet34':
            model_ft = models.resnet34(weights=pretrained_weights)
        elif model_name == 'resnet50':
            model_ft = models.resnet50(weights=pretrained_weights)
        elif model_name == 'resnet101':
            model_ft = models.resnet101(weights=pretrained_weights)
        elif model_name == 'resnet152':
            model_ft = models.resnet152(weights=pretrained_weights)
        else:
            print("Invalid model name, exiting...")
            return None
            
        set_parameter_requires_grad(model_ft, freeze_params)
            
        if new_top:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(weights=pretrained_weights)
        
        set_parameter_requires_grad(model_ft, freeze_params)
            
        if new_top:
            for i in [1,4]:
                in_ftrs = model_ft.classifier[i].in_features
                out_ftrs = model_ft.classifier[i].out_features
                model_ft.classifier[i] = nn.Linear(in_ftrs, out_ftrs)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif "vgg" in model_name:
        """ VGGs
        """
        if model_name == "vgg11":
            model_ft = models.vgg11(weights=pretrained_weights)
        elif model_name == "vgg11_bn":
            model_ft = models.vgg11_bn(weights=pretrained_weights)
        elif model_name == "vgg13":
            model_ft = models.vgg13(weights=pretrained_weights)
        elif model_name == "vgg13_bn":
            model_ft = models.vgg13_bn(weights=pretrained_weights)
        elif model_name == "vgg16":
            model_ft = models.vgg16(weights=pretrained_weights)
        elif model_name == "vgg16_bn":
            model_ft = models.vgg16_bn(weights=pretrained_weights)
        elif model_name == "vgg19":
            model_ft = models.vgg19(weights=pretrained_weights)
        elif model_name == "vgg19_bn":
            model_ft = models.vgg19_bn(weights=pretrained_weights)
        else:
            print("Invalid model name, exiting...")
            return None
        
        set_parameter_requires_grad(model_ft, freeze_params)
        
        if new_top:
            for i in [0,3]:
                in_ftrs = model_ft.classifier[i].in_features
                out_ftrs = model_ft.classifier[i].out_features
                model_ft.classifier[i] = nn.Linear(in_ftrs, out_ftrs)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif "inception" in model_name:
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        if model_name == "inception_v3":
            model_ft = models.inception_v3(weights=pretrained_weights)
        else:
            print("Invalid model name, exiting...")
            return None
            
        set_parameter_requires_grad(model_ft, freeze_params)
        
        if new_top:
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    elif "efficientnet" in model_name:
        """ efficientnets
        """
        
        if model_name == "efficientnet_b0":
            model_ft = models.efficientnet_b0(weights=pretrained_weights)
        elif model_name == "efficientnet_b1":
            model_ft = models.efficientnet_b1(weights=pretrained_weights)
        elif model_name == "efficientnet_b2":
            model_ft = models.efficientnet_b2(weights=pretrained_weights)
        elif model_name == "efficientnet_b3":
            model_ft = models.efficientnet_b3(weights=pretrained_weights)
        elif model_name == "efficientnet_b4":
            model_ft = models.efficientnet_b4(weights=pretrained_weights)
        elif model_name == "efficientnet_b5":
            model_ft = models.efficientnet_b5(weights=pretrained_weights)
        elif model_name == "efficientnet_b6":
            model_ft = models.efficientnet_b6(weights=pretrained_weights)
        elif model_name == "efficientnet_b7":
            model_ft = models.efficientnet_b7(weights=pretrained_weights)
        elif model_name == "efficientnet_v2_l":
            model_ft = models.efficientnet_v2_l(weights=pretrained_weights)
        elif model_name == "efficientnet_v2_m":
            model_ft = models.efficientnet_v2_m(weights=pretrained_weights)
        elif model_name == "efficientnet_v2_s":
            model_ft = models.efficientnet_v2_s(weights=pretrained_weights)
        else:
            print("Invalid model name, exiting...")
            return None
            
        set_parameter_requires_grad(model_ft, freeze_params)
        
        if new_top:
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    elif "mobilenet" in model_name:
        if model_name == 'mobilenet_v2':
            model_ft = models.mobilenet_v2(weights=pretrained_weights)
        elif model_name == 'mobilenet_v3_small':
            model_ft = models.mobilenet_v3_small(weights=pretrained_weights)
        elif model_name == 'mobilenet_v3_large':
            model_ft = models.mobilenet_v3_large(weights=pretrained_weights)
        else:
            print("Invalid model name, exiting...")
            return None
            
        set_parameter_requires_grad(model_ft, freeze_params)
        
        if new_top:
            if model_name == 'mobilenet_v2':
                num_ftrs = model_ft.classifier[1].in_features
                model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
            elif model_name == 'mobilenet_v3_small' or model_name == 'mobilenet_v3_large':
                in_ftrs = model_ft.classifier[0].in_features
                out_ftrs = model_ft.classifier[0].out_features
                model_ft.classifier[0] = nn.Linear(in_ftrs, out_ftrs)
                num_ftrs = model_ft.classifier[3].in_features
                model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)
            
    else:
        print("Invalid model name, exiting...")
        return None

    ## Load new weights for the whole model
    if weights_fn:
        try:
            model.load_state_dict(torch.load(weights_fn))
        except:
            print ("Something went wrong loading weight model")
            model_ft = None
            
    ## If get_features is True remove the classifier layers from the model.
    ## new_top is, of course, dismissed
    if get_features:
        if "vgg" in model_name or "mobilenet" in model_name or "efficientnet" in model_name or "alexnet" in model_name:
            model_ft.classifier = Identity()
        elif "resnet" in model_name:
            model_ft.fc = Identity()
        elif "inception" in model_name:
            model_ft.AuxLogits.fc = Identity()
            model_ft.fc = Identity()
                              
    return model_ft

