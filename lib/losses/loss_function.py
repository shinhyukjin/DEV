import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet as focal_loss
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
import operator

class Hierarchical_Task_Learning:
    def __init__(self,epoch0_loss,stat_epoch_nums=5):
        self.index2term = [*epoch0_loss.keys()]
        self.term2index = {term:self.index2term.index(term) for term in self.index2term}  #term2index
        self.stat_epoch_nums = stat_epoch_nums
        self.past_losses=[]
        self.loss_graph = {#'seg_loss':[],

                           'kd_difi_loss': [],

                           'size2d_loss':[], 
                           'offset2d_loss':[],
                           'offset3d_loss':['size2d_loss','offset2d_loss'], 
                           'size3d_loss':['size2d_loss','offset2d_loss'], 
                           'heading_loss':['size2d_loss','offset2d_loss'], 
                           'depth_loss':['size2d_loss','size3d_loss','offset2d_loss'],
                           'mid_feat_loss':[],
                           'kd_hinton_loss':[],
                           'roi_feature_loss':[]
                           }

    def compute_weight(self,current_loss,epoch):
        T=140
        #compute initial weights
        loss_weights = {}
        eval_loss_input = torch.cat([_.unsqueeze(0) for _ in current_loss.values()]).unsqueeze(0)
        for term in self.loss_graph:
            if len(self.loss_graph[term])==0:
                loss_weights[term] = torch.tensor(1.0).to(current_loss[term].device)
            else:
                loss_weights[term] = torch.tensor(0.0).to(current_loss[term].device) 
        #update losses list
        if len(self.past_losses)==self.stat_epoch_nums:
            past_loss = torch.cat(self.past_losses)
            mean_diff = (past_loss[:-2]-past_loss[2:]).mean(0)
            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff
            c_weights = 1-(mean_diff/self.init_diff).relu().unsqueeze(0)
            
            time_value = min(((epoch-5)/(T-5)),1.0)
            for current_topic in self.loss_graph:
                if len(self.loss_graph[current_topic])!=0:
                    control_weight = 1.0
                    for pre_topic in self.loss_graph[current_topic]:
                        control_weight *= c_weights[0][self.term2index[pre_topic]]      
                    loss_weights[current_topic] = time_value**(1-control_weight)
            #pop first list
            self.past_losses.pop(0)
        self.past_losses.append(eval_loss_input)   
        return loss_weights
    def update_e0(self,eval_loss):
        self.epoch0_loss = torch.cat([_.unsqueeze(0) for _ in eval_loss.values()]).unsqueeze(0)


class GupnetLoss(nn.Module):
    def __init__(self,epoch):
        super().__init__()
        self.stat = {}
        self.epoch = epoch
        self.criterion = nn.MSELoss()


    def forward(self, preds, targets, teacher_pred, task_uncertainties=None):


        kd_loss = self.compute_kd_loss(preds, targets, teacher_pred)
        #kd_difi_loss = self.compute_kd_difficulty_loss_only(preds, targets, teacher_pred)
        #seg_loss = self.compute_segmentation_loss(preds, targets)

        kd_difi_loss = self.compute_kd_difficulty_loss(preds, targets, teacher_pred)
        bbox2d_loss = self.compute_bbox2d_loss(preds, targets)
        bbox3d_loss = self.compute_bbox3d_loss(preds, targets)
        #kd_loss = self.compute_kd_loss(preds, teacher_pred)
        

        #loss = seg_loss + bbox2d_loss + bbox3d_loss + kd_loss
        loss = bbox2d_loss + bbox3d_loss + kd_loss + kd_difi_loss
        #loss = seg_loss + bbox2d_loss + bbox3d_loss + kd_loss + kd_difi_loss

        
        return loss, self.stat


    def compute_segmentation_loss(self, input, target):
        input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss = focal_loss(input['heatmap'], target['heatmap'])
        self.stat['seg_loss'] = loss
        return loss



    # def compute_kd_difficulty_loss(self, input, target, teacher_input):
    #     input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
    #     teacher_input['heatmap'] = torch.clamp(teacher_input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
    #
    #     pos_inds = target['heatmap'].eq(1).float()
    #     neg_inds = target['heatmap'].lt(1).float()
    #
    #     loss = 0
    #
    #     # pos_loss = (1 - input['heatmap']) * pos_inds
    #     pos_loss = torch.log(input['heatmap']) * torch.pow(1 - input['heatmap'], 2) * pos_inds
    #     T_pos_loss = torch.log(teacher_input['heatmap']) * torch.pow(1 - teacher_input['heatmap'], 2) * pos_inds
    #
    #     neg_loss = torch.log(1 - input['heatmap']) * torch.pow(input['heatmap'], 2) * neg_inds
    #     T_neg_loss = torch.log(1 - teacher_input['heatmap']) * torch.pow(teacher_input['heatmap'], 2) * neg_inds
    #     #T_neg_loss = 0
    #
    #     num_pos = pos_inds.float().sum()
    #
    #     a = 0.8
    #
    #     pos_difi_loss = ((a * pos_loss) + ((1-a)*T_pos_loss)).pow(1).sum()
    #     neg_difi_loss = ((a * neg_loss) + ((1-a)*T_neg_loss)).pow(1).sum()
    #
    #     if num_pos == 0:
    #         loss = loss - neg_difi_loss
    #     else:
    #         loss = loss - (pos_difi_loss + neg_difi_loss) / num_pos
    #
    #     self.stat['kd_difi_loss'] = loss * 0.2
    #     return loss


    def compute_kd_difficulty_loss(self, input, target, teacher_input):
        input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        teacher_input['heatmap'] = torch.clamp(teacher_input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)

        pos_inds = target['heatmap'].eq(1).float()
        neg_inds = target['heatmap'].lt(1).float()

        neg_weights = torch.pow(1 - target['heatmap'], 4)

        loss = 0
        a = 0.7
        gamma = 1.5

        # pos_loss = (1 - input['heatmap']) * pos_inds
        pos_loss = torch.log(input['heatmap']) * torch.pow((a*(1 - input['heatmap'])+(1-a)*(1 - teacher_input['heatmap'])), gamma) * pos_inds
        #pos_loss = torch.log(a * input['heatmap'] + (1 - a) * teacher_input['heatmap']) * torch.pow((a * (1 - input['heatmap']) + (1 - a) * (1 - teacher_input['heatmap'])), 2) * pos_inds
        #neg_loss = torch.log(a*(1 - input['heatmap'])+(1-a)*(1 - teacher_input['heatmap'])) * torch.pow(a*input['heatmap']+(1-a)*teacher_input['heatmap'], 2) * neg_inds * neg_weights
        neg_loss = torch.log(1 - input['heatmap']) * torch.pow((a*(input['heatmap'])+(1-a)*(teacher_input['heatmap'])), gamma) * neg_inds * neg_weights

        num_pos = pos_inds.float().sum()

        pos_difi_loss = pos_loss.sum()
        neg_difi_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_difi_loss
        else:
            loss = loss - (pos_difi_loss + neg_difi_loss) / num_pos

        #loss = loss*2

        self.stat['kd_difi_loss'] = loss
        return loss



    def compute_kd_difficulty_loss_only(self, input, target, teacher_input):
        s_logits = F.softmax(input['heatmap'].clone(),dim =1)
        t_logits = F.softmax(teacher_input['heatmap'].clone(), dim=1)

        pos_inds = target['heatmap'].eq(1).float()
        num_pos = pos_inds.float().sum()

        a = 0.7

        if num_pos == 0:
            loss = 0
        else:
            loss = (((a * ((1 - s_logits) * pos_inds) + (1 - a) * ((1 - t_logits) * pos_inds)).pow(2.0).sum()) / num_pos)*0.5

        self.stat['kd_difi_loss'] = loss
        return loss


    def compute_bbox2d_loss(self, input, target):

        if target['mask_2d'].sum() <= 0:
            size2d_loss = torch.tensor(0.0).to(input['size_2d'].device)
            offset2d_loss = torch.tensor(0.0).to(input['size_2d'].device)
        else:
            # compute size2d loss

            size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
            size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
            size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
            # compute offset2d loss
            offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
            offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
            offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')


        loss = offset2d_loss + size2d_loss   
        self.stat['offset2d_loss'] = offset2d_loss
        self.stat['size2d_loss'] = size2d_loss
        return loss


    def compute_bbox3d_loss(self, input, target, mask_type = 'mask_2d'):
        
        if target[mask_type].sum() <= 0:
            depth_loss    = torch.tensor(0.0).to(input['size_3d'].device)
            offset3d_loss = torch.tensor(0.0).to(input['size_3d'].device)
            size3d_loss   = torch.tensor(0.0).to(input['size_3d'].device)
            heading_loss  = torch.tensor(0.0).to(input['size_3d'].device)
        else:
            # compute depth loss
            depth_input = input['depth'][input['train_tag']]
            depth_input, depth_log_variance = depth_input[:, 0:1], depth_input[:, 1:2]
            depth_target = extract_target_from_tensor(target['depth'], target[mask_type])
            depth_loss = laplacian_aleatoric_uncertainty_loss(depth_input, depth_target, depth_log_variance)

            # compute offset3d loss
            offset3d_input = input['offset_3d'][input['train_tag']]
            offset3d_target = extract_target_from_tensor(target['offset_3d'], target[mask_type])
            offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')

            # compute size3d loss
            size3d_input = input['size_3d'][input['train_tag']]
            size3d_target = extract_target_from_tensor(target['size_3d'], target[mask_type])
            size3d_loss = F.l1_loss(size3d_input[:,1:], size3d_target[:,1:], reduction='mean')*2/3+\
                   laplacian_aleatoric_uncertainty_loss(size3d_input[:,0:1], size3d_target[:,0:1], input['h3d_log_variance'][input['train_tag']])/3
            #size3d_loss = F.l1_loss(size3d_input[:,1:], size3d_target[:,1:], reduction='mean')+\
            #       laplacian_aleatoric_uncertainty_loss(size3d_input[:,0:1], size3d_target[:,0:1], input['h3d_log_variance'][input['train_tag']])
            # compute heading loss
            heading_loss = compute_heading_loss(input['heading'][input['train_tag']] ,
                                                target[mask_type],  ## NOTE
                                                target['heading_bin'],
                                                target['heading_res'])
        loss = depth_loss + offset3d_loss + size3d_loss + heading_loss
        
        self.stat['depth_loss'] = depth_loss
        self.stat['offset3d_loss'] = offset3d_loss
        self.stat['size3d_loss'] = size3d_loss
        self.stat['heading_loss'] = heading_loss
        
        
        return loss

    def compute_kd_loss(self, input, target, teacher_input):

        feature_kd_loss = self.criterion(input['feat'],teacher_input['feat'])

        pos_inds = target['heatmap'].eq(1).float()
        num_pos = pos_inds.float().sum()

        pos_inds = pos_inds.sum(dim=1, keepdim=True)

        T = 4
        p_s = F.log_softmax(input['heatmap'] / T, dim=1)*pos_inds
        p_t = F.softmax(teacher_input['heatmap'] / T, dim=1)*pos_inds
        hinton_kd_loss = (nn.KLDivLoss(reduction='sum')(p_s, p_t) * (T ** 2))/num_pos

        roi_feature_loss = (self.criterion(input['roi_feature_masked'], teacher_input['roi_feature_masked'])) * 0.5

        loss = feature_kd_loss + hinton_kd_loss + roi_feature_loss

        self.stat['mid_feat_loss'] = feature_kd_loss
        self.stat['kd_hinton_loss'] = hinton_kd_loss
        self.stat['roi_feature_loss'] = roi_feature_loss

        return loss





### ======================  auxiliary functions  =======================

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask.bool()]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask.bool()]

#compute heading loss two stage style  

def compute_heading_loss(input, mask, target_cls, target_reg):
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    target_cls = target_cls[mask.bool()]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')
    
    # regression loss
    input_reg = input[:, 12:24]
    target_reg = target_reg[mask.bool()]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    return cls_loss + reg_loss    
'''    

def compute_heading_loss(input, ind, mask, target_cls, target_reg):
    """
    Args:
        input: features, shaped in B * C * H * W
        ind: positions in feature maps, shaped in B * 50
        mask: tags for valid samples, shaped in B * 50
        target_cls: cls anns, shaped in B * 50 * 1
        target_reg: reg anns, shaped in B * 50 * 1
    Returns:
    """
    input = _transpose_and_gather_feat(input, ind)   # B * C * H * W ---> B * K * C
    input = input.view(-1, 24)  # B * K * C  ---> (B*K) * C
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    input_cls, target_cls = input_cls[mask], target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    # regression loss
    input_reg = input[:, 12:24]
    input_reg, target_reg = input_reg[mask], target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    return cls_loss + reg_loss
'''


if __name__ == '__main__':
    input_cls  = torch.zeros(2, 50, 12)  # B * 50 * 24
    input_reg  = torch.zeros(2, 50, 12)  # B * 50 * 24
    target_cls = torch.zeros(2, 50, 1, dtype=torch.int64)
    target_reg = torch.zeros(2, 50, 1)

    input_cls, target_cls = input_cls.view(-1, 12), target_cls.view(-1)
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    a = torch.zeros(2, 24, 10, 10)
    b = torch.zeros(2, 10).long()
    c = torch.ones(2, 10).long()
    d = torch.zeros(2, 10, 1).long()
    e = torch.zeros(2, 10, 1)
    print(compute_heading_loss(a, b, c, d, e))

