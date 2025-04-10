from typing import Dict
import torch
from transformers4rec.torch.masking import MaskingInfo

# Copied from Accelerate.
def _pad_across_processes(self, tensor, pad_index=-100):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
    they can safely be gathered.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(self._pad_across_processes(t, pad_index=pad_index) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: self._pad_across_processes(v, pad_index=pad_index) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
        )

    if len(tensor.shape) < 2:
        return tensor
    # Gather all sizes
    size = torch.tensor(tensor.shape, device=tensor.device)[None]
    sizes = self._nested_gather(size).cpu()

    max_size = max(s[1] for s in sizes)
    # When extracting XLA graphs for compilation, max_size is 0,
    # so use inequality to avoid errors.
    if tensor.shape[1] >= max_size:
        return tensor

    # Then pad to the maximum size
    old_size = tensor.shape
    new_size = list(old_size)
    new_size[1] = max_size
    new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
    new_tensor[:, : old_size[1]] = tensor
    return new_tensor

def apply_mask_to_inputs_CLM(
        self,
        inputs: torch.Tensor,
        mask_schema: torch.Tensor,
        training: bool = False,
        testing: bool = False,
    ) -> torch.Tensor:
        if not training and not testing:
            # Replacing the inputs corresponding to padded items with a trainable embedding
            # To mimic training and evaluation masking strategy
            inputs = torch.where(
                mask_schema.unsqueeze(-1).bool(),
                inputs,
                self.masked_item_embedding.to(inputs.dtype),
            )
            return inputs
        
        # # shift sequence of interaction embeddings
        # pos_emb_inp = inputs[:, :-1]
        # # Adding a masked item in the sequence to return to the initial sequence.
        # pos_emb_inp = torch.cat(  # type: ignore
        #     [
        #         pos_emb_inp,
        #         torch.zeros(
        #             (pos_emb_inp.shape[0], 1, pos_emb_inp.shape[2]),
        #             dtype=pos_emb_inp.dtype,
        #         ).to(inputs.device),
        #     ],
        #     axis=1,
        # )

        pos_emb_inp = inputs[:, :]
        pos_emb_inp_new = pos_emb_inp.clone()
        # Iterate over each row in the boolean tensor
        for i in range(mask_schema.shape[0]):
            # Find the index of the last True value in the row
            # If there's no True value, idx will be -1
            idx = (mask_schema[i].nonzero(as_tuple=True)[0]).max().item() if mask_schema[i].any() else -1
            # Replace corresponding item in feature tensor with a zero matrix
            if idx != -1:
                pos_emb_inp_new[i, idx] = torch.zeros(pos_emb_inp.shape[2], dtype=pos_emb_inp.dtype).to(inputs.device)

        pos_emb_inp = pos_emb_inp_new
        # Replacing the inputs corresponding to padded items with a trainable embedding
        pos_emb_inp = torch.where(
            mask_schema.unsqueeze(-1).bool(),
            pos_emb_inp,
            self.masked_item_embedding.to(pos_emb_inp.dtype),
        )
        return pos_emb_inp


def _compute_masked_targets_mask_last_item(
        self, item_ids: torch.Tensor, training: bool = False, testing: bool = False, additional_cat_ids: Dict[str, torch.Tensor] = None
    ) -> MaskingInfo:
        if not training and not testing:
            mask_labels = item_ids != self.padding_idx
            # new for t5
            self.decoder_target_sequence = item_ids
            return MaskingInfo(mask_labels, item_ids, additional_targets_labels=additional_cat_ids)

        masking_info = self.predict_all(item_ids)
        mask_labels, labels = masking_info.schema, masking_info.targets
        # new for t5
        self.decoder_target_sequence = labels

        # get the additional category labels
        if additional_cat_ids is not None:
            for key, cat_ids in additional_cat_ids.items():
                masking_info = self.predict_all(cat_ids)
                additional_cat_ids[key] = masking_info.targets

        if (self.eval_on_last_item_seq_only and not training) or (
            self.train_on_last_item_seq_only and training
        ):
            rows_ids = torch.arange(
                labels.size(0), dtype=torch.long, device=item_ids.device  # type: ignore
            )
            last_item_sessions = mask_labels.sum(dim=1) - 1
            label_seq_trg_eval = torch.zeros(
                labels.shape, dtype=labels.dtype, device=item_ids.device
            )
            label_seq_trg_eval[rows_ids, last_item_sessions] = labels[rows_ids, last_item_sessions]
            # Updating labels and mask
            labels = label_seq_trg_eval
            # old
            # mask_labels = label_seq_trg_eval != self.padding_idx
            # # We only mask padded positions
            # mask_labels = item_ids != self.padding_idx

            # new also apply same masking strategy to the additional category labels
            # get the additional category labels
            if additional_cat_ids is not None:
                for key, cat_ids in additional_cat_ids.items():
                    cat_label_seq_trg_eval = torch.zeros(labels.shape, dtype=labels.dtype, device=item_ids.device)
                    cat_label_seq_trg_eval[rows_ids, last_item_sessions] = cat_ids[rows_ids, last_item_sessions]
                    additional_cat_ids[key] = cat_label_seq_trg_eval
            # ----------------------------  end of new --------------------------------

        return MaskingInfo(mask_labels, labels, additional_targets_labels=additional_cat_ids)

