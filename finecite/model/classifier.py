import torch
from torch.nn import Parameter, CrossEntropyLoss
from torch import nn


class ClsClassifier(torch.nn.Module):
    def __init__(self, args, config):
        self.dtype = args.dtype
        self.cls_type = args.cls_type
        
        super().__init__()
        self.dropout = torch.nn.Dropout(args.dropout)
        
        if args.cls_type == 'linear':
            self.pre_cls = torch.nn.Linear(config.hidden_size * 3, config.hidden_size, dtype=args.dtype).to(args.device)
        if args.cls_type in ['inf', 'perc', 'back']:
            self.pre_cls_abl = torch.nn.Linear(config.hidden_size * 2, config.hidden_size, dtype=args.dtype).to(args.device)
        self.cls = torch.nn.Linear(config.hidden_size, args.num_labels, dtype=args.dtype).to(args.device)
        

    def _extract_cont_emb(self, hs, tok_lbl):
        res = []
        for i in range(1,4):
            denom = torch.sum(tok_lbl == i, -1, keepdim=True) + 1e-07
            feat = torch.sum(hs * (tok_lbl ==i).unsqueeze(-1), dim=1) / denom 
            res.append(feat.to(self.dtype))
        mask = (tok_lbl==1).to(torch.int) | (tok_lbl==2).to(torch.int) | (tok_lbl==3).to(torch.int)
        denom = torch.sum(mask, -1, keepdim=True) + 1e-07
        tok_mean = torch.sum(hs * (mask).unsqueeze(-1), dim=1) / denom 
        return res, tok_mean

    def _process_hidden_states(self, hidden_states, tok_lbl):
        (inf_emb, perc_emb, back_emb), total_mean = self._extract_cont_emb(hidden_states, tok_lbl)
        match self.cls_type:
            case 'balanced':
                cls_emb = torch.mean(torch.stack((inf_emb, perc_emb,back_emb)), dim=0)
            case 'total':
                cls_emb = total_mean
            case 'linear':
                cls_emb = torch.cat((inf_emb, perc_emb, back_emb),dim=1)
                cls_emb = self.pre_cls(cls_emb)
            case 'inf':
                cls_emb = torch.cat((perc_emb, back_emb),dim=1)
                cls_emb = self.pre_cls_abl(cls_emb)
            case 'perc':
                cls_emb = torch.cat((inf_emb, back_emb),dim=1)
                cls_emb = self.pre_cls_abl(cls_emb)
            case 'back':
                cls_emb = torch.cat((inf_emb, perc_emb),dim=1)
                cls_emb = self.pre_cls_abl(cls_emb)
            case _:
                raise NotImplementedError()
        return cls_emb

    def forward(self, hidden_states, tok_lbl):        
        cls_emb = self._process_hidden_states(hidden_states, tok_lbl)
        pre_out = self.dropout(cls_emb)
        out = self.cls(pre_out)
        return out
    
class ExtClassifier(torch.nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.ext_type = args.ext_type
        self.dropout = torch.nn.Dropout(args.dropout)
        self.cls = torch.nn.Linear(config.hidden_size, args.num_labels, dtype=args.dtype).to(args.device)
        if args.ext_type in ['bilstm', 'bilstm_crf']:
            self.lstm = torch.nn.LSTM(config.hidden_size, config.hidden_size // 2, num_layers=1, bidirectional=True, dtype=args.dtype, batch_first=True).to(args.device)

    def forward(self, hidden_state):
        pre_out = self.dropout(hidden_state)
        if self.ext_type in ['bilstm', 'bilstm_crf']:
            pre_out, self.hidden = self.lstm(pre_out)
        out = self.cls(pre_out)
        if self.ext_type in ['linear', 'bilstm']:
            out = torch.transpose(out, 1, 2)
        return out
    


# implementation based on https://medium.com/towards-data-science/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea
class CRF(nn.Module):
    """
    Linear-chain Conditional Random Field (CRF).

    Args:
        nb_labels (int): number of labels in your tagset, including special symbols.
        bos_tag_id (int): integer representing the beginning of sentence symbol in
            your tagset.
        eos_tag_id (int): integer representing the end of sentence symbol in your tagset.
        pad_tag_id (int, optional): integer representing the pad symbol in your tagset.
            If None, the model will treat the PAD as a normal tag. Otherwise, the model
            will apply constraints for PAD transitions.
        batch_first (bool): Whether the first dimension represents the batch dimension.
    """

    def __init__(self, num_labels, device, first_tok_idx = None):
        super().__init__()
        self.nb_labels = num_labels + 2
        self.BOS_TAG_ID = num_labels
        self.EOS_TAG_ID = num_labels + 1
        self.device = device
        self.first_token_idx = first_tok_idx

        self.transitions = nn.Parameter(torch.randn(self.nb_labels, self.nb_labels, requires_grad=True).to(self.device))
        self.init_weights()
        
    def init_weights(self):
        # no transitions allowed to the beginning of sentence
        self.transitions.data[:, self.BOS_TAG_ID] = -10000.0
        # no transition alloed from the end of sentence
        self.transitions.data[self.EOS_TAG_ID, :] = -10000.0

    def forward(self, emissions, tags, mask=None):
        nll = -self.log_likelihood(emissions, tags, mask=mask)
        return nll

    def log_likelihood(self, emissions, tags, mask=None):
        """Compute the probability of a sequence of tags given a sequence of
        emissions scores.

        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape of (batch_size, seq_len, nb_labels) if batch_first is True,
                (seq_len, batch_size, nb_labels) otherwise.
            tags (torch.LongTensor): Sequence of labels.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.

        Returns:
            torch.Tensor: the log-likelihoods for each sequence in the batch.
                Shape of (batch_size,)
        """
        # add bos and eos to emissions
        size = emissions.size()
        emissions = torch.cat((emissions, torch.full((size[0], size[1], 2), -100).to(self.device)), dim=2)

        if self.first_token_idx:
            emissions = emissions[:, self.first_token_idx:]
            tags = tags[:, self.first_token_idx:]
        
        if mask is None:
            mask = tags != -100
            tags[~mask] = 0
            
        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        return torch.mean(scores - partition) / 100
    
    def predict(self, emissions, mask=None):
        """Find the most probable sequence of labels given the emissions using
        the Viterbi algorithm.

        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape (batch_size, seq_len, nb_labels) if batch_first is True,
                (seq_len, batch_size, nb_labels) otherwise.
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.

        Returns:
            torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists: the best viterbi sequence of labels for each batch.
        """
        # add bos and eos to emissions
        size = emissions.size()
        emissions = torch.cat((emissions, torch.full((size[0], size[1], 2), -100).to(self.device)), dim=2)
        
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool).to(self.device)
            
        if self.first_token_idx:
            emissions = emissions[:, self.first_token_idx:]
            mask = mask[:, self.first_token_idx:]
        scores, sequences = self._viterbi_decode(emissions, mask)
        return sequences

    def decode(self, emissions, mask=None):
        sequences = self.predict(emissions, mask)
        sequences = torch.tensor([l for labels in sequences for l in labels]).to(self.device)
        return sequences

    def _compute_scores(self, emissions, tags, mask):
        """Compute the scores for a given batch of emissions with their tags.

        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            tags (Torch.LongTensor): (batch_size, seq_len)
            mask (Torch.FloatTensor): (batch_size, seq_len)

        Returns:
            torch.Tensor: Scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size).to(self.device)

        # save first and last tags to be used later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()

        # add the transition from BOS to the first tags for each batch
        t_scores = self.transitions[self.BOS_TAG_ID, first_tags]

        # add the [unary] emission scores for the first tags for each batch
        # for all batches, the first word, see the correspondent emissions
        # for the first tags (which is a list of ids):
        # emissions[:, 0, [tag_1, tag_2, ..., tag_nblabels]]
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()
        
        # the scores for a word is just the sum of both scores
        scores += e_scores + t_scores

        # now lets do this for each remaining word
        for i in range(1, seq_length):

            # we could: iterate over batches, check if we reached a mask symbol
            # and stop the iteration, but vecotrizing is faster due to gpu,
            # so instead we perform an element-wise multiplication
            is_valid = mask[:, i]

            previous_tags = tags[:, i - 1]
            current_tags = tags[:, i]

            # calculate emission and transition scores as we did before
            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[previous_tags, current_tags]

            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid
            
            scores += e_scores + t_scores



        # add the transition from the end tag to the EOS tag for each batch
        scores += self.transitions[last_tags, self.EOS_TAG_ID]

        return scores

    def _compute_log_partition(self, emissions, mask):
        """Compute the partition function in log-space using the forward-algorithm.

        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)

        Returns:
            torch.Tensor: the partition scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # in the first iteration, BOS will have all the scores
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

        for i in range(1, seq_length):
            # (bs, nb_labels) -> (bs, 1, nb_labels)
            e_scores = emissions[:, i].unsqueeze(1)

            # (nb_labels, nb_labels) -> (bs, nb_labels, nb_labels)
            t_scores = self.transitions.unsqueeze(0)

            # (bs, nb_labels)  -> (bs, nb_labels, 1)
            a_scores = alphas.unsqueeze(2)

            scores = e_scores + t_scores + a_scores
                
            new_alphas = torch.logsumexp(scores, dim=1)

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (~is_valid) * alphas

        # add the scores for the final transition
        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        # return a *log* of sums of exps
        return torch.logsumexp(end_scores, dim=1)

    def _viterbi_decode(self, emissions, mask):
        """Compute the viterbi algorithm to find the most probable sequence of labels
        given a sequence of emissions.

        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)

        Returns:
            torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists of ints: the best viterbi sequence of labels for each batch
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # in the first iteration, BOS will have all the scores and then, the max
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

        backpointers = []

        for i in range(1, seq_length):
            # (bs, nb_labels) -> (bs, 1, nb_labels)
            e_scores = emissions[:, i].unsqueeze(1)

            # (nb_labels, nb_labels) -> (bs, nb_labels, nb_labels)
            t_scores = self.transitions.unsqueeze(0)

            # (bs, nb_labels)  -> (bs, nb_labels, 1)
            a_scores = alphas.unsqueeze(2)

            # combine current scores with previous alphas
            scores = e_scores + t_scores + a_scores

            # so far is exactly like the forward algorithm,
            # but now, instead of calculating the logsumexp,
            # we will find the highest score and the tag associated with it
            max_scores, max_score_tags = torch.max(scores, dim=1)

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * max_scores + (~is_valid) * alphas

            # add the max_score_tags for our list of backpointers
            # max_scores has shape (batch_size, nb_labels) so we transpose it to
            # be compatible with our previous loopy version of viterbi
            backpointers.append(max_score_tags.t())

        # add the scores for the final transition
        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        # get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):

            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences

    def _find_best_path(self, sample_id, best_tag, backpointers):
        """Auxiliary function to find the best path sequence for a specific sample.

            Args:
                sample_id (int): sample index in the range [0, batch_size)
                best_tag (int): tag which maximizes the final score
                backpointers (list of lists of tensors): list of pointers with
                shape (seq_len_i-1, nb_labels, batch_size) where seq_len_i
                represents the length of the ith sample in the batch

            Returns:
                list of ints: a list of tag indexes representing the bast path
        """

        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):

            # recover the best_tag at this timestep
            best_tag = backpointers_t[best_tag][sample_id].item()

            # append to the beginning of the list so we don't need to reverse it later
            best_path.insert(0, best_tag)

        return best_path
