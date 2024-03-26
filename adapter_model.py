import torch

class AdapterModel:
    # hard code for the adapter
    def __init__(self):

        self.adapter_weight = [
                                    'transformer.encoder.layers.0.adapter_encoder.dense_h_to_4h.weight',
                                    'transformer.encoder.layers.0.adapter_encoder.dense_4h_to_h.weight',
                                    'transformer.encoder.layers.27.adapter_encoder.dense_h_to_4h.weight',
                                    'transformer.encoder.layers.27.adapter_encoder.dense_4h_to_h.weight']

        self.prefix_weight = [

        ]

        self.copy_weights_source = [
                                    'transformer.encoder.layers.0.mlp.dense_h_to_4h.weight',
                                    'transformer.encoder.layers.0.mlp.dense_4h_to_h.weight',
                                    'transformer.encoder.layers.27.mlp.dense_h_to_4h.weight',
                                    'transformer.encoder.layers.27.mlp.dense_4h_to_h.weight']

    def set_adapter_weights(self, model):

        with torch.no_grad():
            adapter_name = self.adapter_weight
            copy_weights_name = self.copy_weights_source

            # for target_name, source_name in zip(adapter_name, copy_weights_name):
            #     target_param = model.get_parameter(target_name)
            #     torch.nn.init.trunc_normal_(target_param)
            #     #source_param = model.get_parameter(source_name)
            #     #target_param.copy_(source_param)
            #
            # self.adapter_weight.append('transformer.prefix_encoder.embedding.weight')
            for name, param in model.named_parameters():
                if 'adapter_encoder' in name or "prefix" in name:
                    torch.nn.init.trunc_normal_(param)
                    # print(param.dtype)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                # if name not in self.adapter_weight :
                #
                #     param.requires_grad = False
                # else:
                #     param.requires_grad = True
                    # print(name)
