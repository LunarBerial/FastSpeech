import torch
import torch.nn as nn
from torch.nn import functional as F
from UNetModel_torch import U_Net

class Attention(nn.Module):
	
	def __init__(self, units, prefix):
		super (Attention, self).__init__ (prefix = prefix)
		
		self.units = units
		
	
	def forward(self, key, query, embedding, mask_enc, sp_type_lists, silence_time, ctx):

		atten = torch.bmm (query, key.transpose(1, 2)) / np.sqrt (self.units)

        mask_enc = mask_enc.unsqueeze (1)

        atten = torch.sub (atten, (1.0 - mask_enc) * 1e5)

        attenMax = torch.max (atten, axis = 2, keepdim = True)[0]

        index = atten >= attenMax
        
        index = index.type_as(embedding)

        return torch.bmm (index, embedding), index
        
        
class AddRNN (nn.Module):

    def __init__ (self, prefix):

        super (AddRNN, self).__init__ (prefix = prefix)
    
    def reset(self, x):

        self.output = torch.zeros_like(x)

    def forward (self, x, T):
    
        for i in range(T):
            self.output[:,i] = x[:,i,:] + self.output[:,i-1,:]

        return self.output
        
        
class DynamicPosEnc (nn.Module):

    def __init__ (self, prefix):

        super (DynamicPosEnc, self).__init__ (prefix = prefix)

        self.Add_RNN = AddRNN ('Add_RNN_')

    def forward (self, r, values_L, T):

        self.Add_RNN.reset ()

        values_T = self.Add_RNN(T, r)

        values_T = values_T - 0.5 * r

        values_TL = torch.bmm (values_T, values_L.transpose(1, 2))

        values_sin = torch.sin (values_TL)

        values_cos = torch.cos (values_TL)

        return torch.cat ((values_sin, values_cos) , dim = 2)  
        

class Relative (nn.Module):

    def __init__ (self, prefix):

        super (Relative, self).__init__ (prefix = prefix)

        self.add_rnn = AddRNN ()

    def forward (self, index, mask_enc, mask_dec, x_T, y_index):

        self.add_rnn.reset ()

        mask_dec = mask_dec.unsqueeze (2)

        rate = torch.sum (index * mask_dec, dim = 1, keepdims = False)

        rate = rate.unsqueeze (2)

        rateSum = self.add_rnn(x_T, rate)[0]

        rateSum = rateSum * mask_enc.unsqueeze ( axis = 2)

        rateSum_ex = rateSum.transpose (1, 2)

        y_index_ex = y_index.unsqueeze ( 2)

        minus = y_index_ex - rateSum_ex

        minus_extract = - (minus >= 0).type_as(minus) * 1e8 + minus
        
        shift = torch.max (minus_extract, axis = 2, keepdim = True)[0]

        rate_per_frame = torch.bmm (index, rate)

        fraction = (shift + rate_per_frame) / rate_per_frame

        relative_posenc_1 = torch.exp(- 0.5 * torch.mul(1.5 * fraction - 0.75, 1.5 * fraction - 0.75) / 0.16)
        relative_posenc_2 = torch.exp(- 0.5 * torch.mul(0.75 * fraction - 0.75, 0.75 * fraction - 0.75) / 0.16)
        relative_posenc_3 = torch.exp(- 0.5 * torch.mul(0.75 * fraction, 0.75 * fraction) / 0.16)

        return torch.cat((relative_posenc_1, relative_posenc_2, relative_posenc_3), dim=2) * mask_dec


class End2End_1 (nn.Module):

    def __init__ (self, num_id, dim_id, num_phoneme, num_tone, num_order, num_seg, num_down_enc, num_down_dec, dim_embed_phoneme, dim_embed_tone, dim_embed_order,dim_embed_seg, size_enc, size_dec, size_output, dp_dec, prefix):

        super (End2End_1, self).__init__ (prefix = prefix)

        self.num_id = num_id
        self.dim_id = dim_id
        self.num_phoneme = num_phoneme
        self.num_tone = num_tone
        self.num_order = num_order
        self.num_seg = num_seg
        self.num_down_enc = num_down_enc
        self.num_down_dec = num_down_dec
        self.dim_embed_phoneme = dim_embed_phoneme
        self.dim_embed_seg = dim_embed_seg
        self.dim_embed_tone = dim_embed_tone
        self.dim_embed_order = dim_embed_order
        self.size_enc = size_enc
        self.size_dec = size_dec
        self.size_output = size_output
        self.dp_dec = dp_dec  

        

        self.emd_id = nn.Embedding (self.num_id, self.dim_id) 

        self.emd_phoneme = nn.Embedding (self.num_phoneme, self.dim_embed_phoneme, ) 
      
        self.emd_tone = nn.Embedding (self.num_tone, self.dim_embed_tone)

        self.emd_order = nn.Embedding (self.num_order, self.dim_embed_order)


        self.emd_seg = nn.Linear (self.num_seg, self.dim_embed_seg)

        self.emd_seg_rate = nn.Linear (self.num_seg, self.dim_embed_seg)


        self.emd_phoneme_rate = nn.Embedding (self.num_phoneme, self.dim_embed_phoneme) 

        self.emd_tone_rate = nn.Embedding (self.num_tone, self.dim_embed_tone)

        self.emd_order_rate = nn.Embedding (self.num_order, self.dim_embed_order)
        
        dense_enc_input_len = self.dim_embed_phoneme + self.dim_embed_tone + dim_embed_seg + dim_embed_order

        self.U_Net_rate = U_Net (dense_enc_input_len, self.num_id, self.dim_id, 0, 256, 1, self.num_down_enc, [self.size_enc] * self.num_down_enc, [1] * self.num_down_enc, torch.tanh, 'U_Net_rate_', dropout = 0.0)        
        
        self.dense_enc_1 = nn.Linear (dense_enc_input_len, self.size_enc)

        self.id_bias_1 = nn.Linear (self.size_enc, self.size_enc, use_bias = False)

        self.dense_enc_2 = nn.Linear (self.size_enc, self.size_enc)

        self.id_bias_2 = nn.Linear (self.size_enc, self.size_enc, use_bias = False)

        self.dense_enc_3 = nn.Linear (self.size_enc, self.size_enc)
  
        self.id_bias_3 = nn.Linear (self.size_enc, self.size_enc, use_bias = False)

        self.Atten = Attention (self.size_dec, 'atten_')

        self.Dynamic = DynamicPosEnc ('DynamicPosEnc_')

        self.rel = Relative ('rel_')

        self.dense_dec = nn.Linear (515, self.size_dec)

        self.dense_dec_id_bias = nn.Linear (self.dim_id, self.size_dec, use_bias = False)  

        self.U_Net_dec = U_Net (self.num_id, self.dim_id, 0, 256, self.size_output, self.num_down_dec, [self.size_dec] * self.num_down_dec, [1] * self.num_down_dec, torch.tanh, 'U_Net_dec_', dropout = self.dp_dec)



    def forward (self, F, p, t, o, T, dy_dec, values_L, mask_enc, ratio, mask_dec,seg, id, y_index, sp_type_lists, silence_time, ctx):
        id_emd = self.emd_id (id)

        p_embed = self.emd_phoneme (p)
        p_embed_rate = self.emd_phoneme_rate (p)
        t_embed = self.emd_tone (t)
        t_embed_rate = self.emd_tone_rate (t)
        o_embed = self.emd_order (o)
        o_embed_rate = self.emd_order_rate (o)


        s_embed = self.emd_seg (seg)
        s_embed_rate = self.emd_seg_rate (seg)

        x_embed = torch.cat ((p_embed, t_embed,s_embed, o_embed), dim = 2)  
        x_embed_rate = torch.cat ((p_embed_rate, t_embed_rate, s_embed_rate,o_embed_rate), dim = 2)

        ratio = torch.unsqueeze (torch.unsqueeze (ratio, axis = 1), axis = 2)

        rate_x = self.U_Net_rate (x_embed_rate * torch.unsqueeze (mask_enc, axis = 2), id)

        rate_x = torch.relu (rate_x + ratio)
        rate_x[rate_x < 2] = 2
        rate_x[rate_x > 39] = 39
        
        enc = torch.tanh ((self.dense_enc_1 (x_embed) +  torch.unsqueeze (self.id_bias_1 (id_emd), axis = 1)) * torch.unsqueeze (mask_enc, axis = 2))

        enc = torch.tanh ((self.dense_enc_2 (enc) + torch.unsqueeze (self.id_bias_2 (id_emd), axis = 1)) * torch.unsqueeze (mask_enc, axis = 2))

        enc = torch.tanh ((self.dense_enc_3 (enc) + torch.unsqueeze (self.id_bias_3 (id_emd), axis = 1)) * torch.unsqueeze (mask_enc, axis = 2))

        dy_enc = self.Dynamic (rate_x, values_L, T)

        attention, atten_index = self.Atten (dy_enc, dy_dec, enc, mask_enc, sp_type_lists, silence_time, ctx)  #

        rel_info = self.rel (atten_index, mask_enc, mask_dec, T, y_index)

        mask_dec = torch.unsqueeze (mask_dec, axis = 2)    
 
        #temp = F.concat (attention, rel_info, dim = 2) 

        #temp = self.dense_dec (temp) + F.expand_dims (self.dense_dec_id_bias (id_emd), axis = 1)
        #a,b=self.U_Net_dec (temp * mask_dec, id)
        #print (rate_x * mask_enc).shape,mask_enc.shape,F.sum (rate_x * mask_enc, axis = [1, 2])
        #return self.U_Net_dec (temp * mask_dec, id),F.sum (rate_x * mask_enc, axis = [1, 2])#, F.sum (rate_x * mask_enc, axis = [1, 2])#rate_x
 
        mask_enc = torch.unsqueeze (mask_enc, axis = 2)

        return attention, torch.sum (rate_x * mask_enc, axis = [1, 2]), atten_index, id_emd, rate_x
