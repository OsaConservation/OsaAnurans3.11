import torch
from avex import load_model
from bacpipe.model_pipelines.model_utils import ModelBaseClass
from torch import nn

BATCH_SIZE = 1  # Keep at 1 initially; increase after testing
SAMPLE_RATE = 16000
LENGTH_IN_SAMPLES = 16000 * 1  # 3 seconds


class AvexModel(ModelBaseClass, nn.Module):
    def __init__(self, device=None, **kwargs):
        """
        Initialize the AVEX AVES2 BEATs model.

        Parameters
        ----------
        device : str, optional
            Device used for inference ("cuda" or "cpu").
        **kwargs
            Additional arguments passed to ModelBaseClass.
        """

        super().__init__(sr=SAMPLE_RATE, segment_length=LENGTH_IN_SAMPLES, **kwargs)

        nn.Module.__init__(self)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        # Load AVEX AVES2 BEATs model
        self.model = load_model(
            "esp_aves2_sl_beats_all", return_features_only=True, device=str(self.device)
        )

        self.model.eval()

    def preprocess(self, audio):
        """
        Pass audio through unchanged.
        """
        return audio

    @torch.inference_mode()
    def __call__(self, input):
        """
        Extract AVEX embeddings and mean-pool temporal features.

        Parameters
        ----------
        input : torch.Tensor
            Input audio, shape (batch, samples).

        Returns
        -------
        torch.Tensor
            Mean-pooled embeddings, shape (batch, 768).
        """

        embeds = []

        for batch in input.split(BATCH_SIZE):
            batch = batch.to(self.device)

            # AVEX output: (batch, 48, 768)
            out = self.model(batch)

            # Mean pooling over temporal dimension
            out = out.mean(dim=1)

            embeds.append(out)

        return torch.cat(embeds, dim=0)
