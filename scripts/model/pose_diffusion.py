import torch
import torch.nn as nn

from .diffusion_net import *
from .diffusion_util import *
from model.tcn import TemporalConvNet

class PoseDiffusion(nn.Module):
    def __init__(self, args, lang_model=None):
        super().__init__()
        self.pre_length = args.n_pre_poses
        self.gen_length = args.n_poses - args.n_pre_poses
        self.input_context = args.input_context

        # add attribute args for sampling
        self.args = args
        pose_dim = args.pose_dim
        diff_hidden_dim = args.diff_hidden_dim
        block_depth = args.block_depth

        audio_feat_dim = 32
        wordembed_dim = 32
        if args.input_context == 'audio':
            add_feat = audio_feat_dim
        elif args.input_context == 'both' and lang_model is not None:
            self.text_encoder = TextEncoderTCN(args, lang_model.n_words,
                                               args.wordembed_dim, 
                                               pre_trained_embedding=lang_model.word_embedding_weights,
                                               dropout=args.dropout_prob)
            add_feat = audio_feat_dim + wordembed_dim
        
        self.in_size = add_feat + pose_dim + 1 
        self.audio_encoder = WavEncoder()

        self.classifier_free = args.classifier_free
        if self.classifier_free:
            self.null_cond_prob = args.null_cond_prob
            self.null_cond_emb = nn.Parameter(torch.randn(1, self.in_size))

        self.diffusion_net = DiffusionNet(
            net = TransformerModel( num_pose=args.n_poses,
                                    pose_dim=pose_dim, 
                                    embed_dim=pose_dim+3+self.in_size,
                                    hidden_dim=diff_hidden_dim,
                                    depth=block_depth//2,
                                    decoder_depth=block_depth//2
                                    ),
            var_sched = VarianceSchedule(
                num_steps=500,
                beta_1=1e-4,
                beta_T=0.02,
                mode='linear'
            )
        )

    def get_loss(self, x, pre_seq, in_audio, in_text=None):

        if self.input_context == 'audio':
            audio_feat_seq = self.audio_encoder(in_audio) # output (bs, n_frames, feat_size)
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        elif self.input_context == 'both':
            audio_feat_seq = self.audio_encoder(in_audio)
            text_feat_seq, _ = self.text_encoder(in_text)
            in_data = torch.cat((pre_seq, audio_feat_seq, text_feat_seq), dim=2)
        else:
            assert False

        if self.classifier_free:
            mask = torch.zeros((x.shape[0],), device = x.device).float().uniform_(0, 1) < self.null_cond_prob
            in_data = torch.where(mask.unsqueeze(1).unsqueeze(2), self.null_cond_emb.repeat(in_data.shape[1],1).unsqueeze(0), in_data)

        neg_elbo = self.diffusion_net.get_loss(x, in_data)

        return neg_elbo
        
    def sample(self, pose_dim, pre_seq, in_audio, in_text=None):

        if self.input_context == 'audio':
            audio_feat_seq = self.audio_encoder(in_audio)
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        elif self.input_context == 'both':
            audio_feat_seq = self.audio_encoder(in_audio)
            text_feat_seq, _ = self.text_encoder(in_text)
            in_data = torch.cat((pre_seq, audio_feat_seq, text_feat_seq), dim=2)

        if self.classifier_free:
            uncondition_embedding = self.null_cond_emb.repeat(in_data.shape[1],1).unsqueeze(0)
            samples = self.diffusion_net.sample(self.args.n_poses, in_data, pose_dim, uncondition_embedding=uncondition_embedding)
        else:
            samples = self.diffusion_net.sample(self.args.n_poses, in_data, pose_dim)
        return samples

class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=6),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=6),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=6),
        )

    def forward(self, wav_data):
        
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)  # to (batch x seq x dim)

class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py 
        code from https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context
    """
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()

        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=args.freeze_wordembed)
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        num_channels = [args.hidden_size] * args.n_layers
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], 32)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0