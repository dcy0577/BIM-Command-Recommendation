# Transformer4rec
This is a customized version of the original t4rec repo https://github.com/NVIDIA-Merlin/Transformers4Rec. The changes are documented in `model/README.md`.

Some helpful visuals for understanding the masking strategies used in the model:

Causal Language Modeling (CLM):
!["CLM"](CLM.png)

Masked Language Modeling (MLM):
!["MLM"](MLM.png)

## Masking in different types of Transformers
Decoder-only
![clm_decoder](clm_decoder.png)

Encoder-Decoder
![mlm_enco_deco](mlm_encoder_decoder.png)

Encoder-only
![mlm_encoder](mlm_encoder.png)
