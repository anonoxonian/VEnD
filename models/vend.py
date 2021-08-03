from transformers import LongformerConfig, LongformerForSequenceClassification, LongformerModel
from transformers.models.longformer.modeling_longformer import LongformerClassificationHead
import torch
from torch.nn.functional import softmin

class VEND(LongformerForSequenceClassification):
    def __init__(self, output_dim, ss_index, num_neighbours):
        config = LongformerConfig()
        config.num_labels = output_dim
        config.output_attentions = True
        super(LongformerForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.classifier = LongformerClassificationHead(config)
        self.init_weights()
        self.ss_index = ss_index
        self.num_neighbours = num_neighbours

    def forward(self, *args, **kwargs):
        longformer_output = super().forward(*args, **kwargs)
        position = longformer_output.logits
        # similarity search on position
        D, I = self.ss_index.search(position.cpu(), self.num_neighbours)
        # for batch size b, and n = self.num_neighbours, D and I are b*n arrays
        # D is the squared distances
        distances = torch.Tensor(D, requires_grad=True).sqrt()
        scores = softmin(distances, dim=1)
        return {"entities": I, "positions": position, "scores": scores}