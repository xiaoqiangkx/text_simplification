import numpy as np
from util import constant
from util.sari import SARIsent, WeightedSARIsent
from util.fkgl import get_fkgl
from model.lm import GoogleLM

from nltk.translate.bleu_score import sentence_bleu


class Metric:
    def __init__(self, model_config, data):
        self.model_config = model_config
        self.data = data

    def truncate_sent(self, sent, bos=3, eos=4):
        sent = list(sent)
        if eos in sent:
            eos = sent.index(eos)
            sent = sent[:eos]
        s_i = 0
        if len(sent) > 0 and sent[s_i] == bos:
            while s_i + 1 < len(sent) and sent[s_i + 1] == bos:
                s_i += 1
            sent = sent[s_i + 1:]
        return sent

    def self_crititcal_reward(self, sample_target_list, greed_target_list, gt_simp_list, gt_comp_list,
                              rule_target_input_placeholder):
        rewards = []
        batch_size = np.shape(gt_simp_list)[0]
        num_steps = np.shape(gt_simp_list)[1]
        for batch_i in range(batch_size):
            reward = [1.0 for _ in range(num_steps)]
            cur_sample_target_list = self.truncate_sent(sample_target_list[batch_i])
            cur_greed_target_list = self.truncate_sent(greed_target_list[batch_i])
            cur_gt_simp_list = self.truncate_sent(gt_simp_list[batch_i])
            cur_gt_comp_list = self.truncate_sent(gt_comp_list[batch_i])

            if 'dummy' in self.model_config.rl_configs:
                sample_sent = [self.data.vocab_simple.describe(w) for w in cur_sample_target_list]
                greed_sent = [self.data.vocab_simple.describe(w) for w in cur_greed_target_list]
                if 'zhao' in sample_sent:
                    reward_sample = 1.0
                else:
                    reward_sample = 0.0
                if 'zhao' in greed_sent:
                    reward_greed = 1.0
                else:
                    reward_greed = 0.0

                reward = [r * (reward_sample-reward_greed) for r in reward]

            if 'sari' in self.model_config.rl_configs:
                sari_weight = self.model_config.rl_configs['sari_weight']
                cur_sample_target_str = ' '.join([str(o) for o in cur_sample_target_list])
                cur_greed_target_str = ' '.join([str(o) for o in cur_greed_target_list])
                cur_gt_simp_str = ' '.join([str(o) for o in cur_gt_simp_list])
                cur_gt_comp_str = ' '.join([str(o) for o in cur_gt_comp_list])
                try:
                    reward_sample = sari_weight * SARIsent(cur_gt_comp_str, cur_sample_target_str, [cur_gt_simp_str]) + (1-sari_weight) * SARIsent(cur_gt_comp_str, cur_gt_simp_str, [cur_sample_target_str])
                except ZeroDivisionError:
                    reward_sample = 0.0
                try:
                    reward_greed = sari_weight * SARIsent(cur_gt_comp_str, cur_greed_target_str, [cur_gt_simp_str]) + (1-sari_weight) * SARIsent(cur_gt_comp_str, cur_gt_simp_str, [cur_greed_target_str])
                except ZeroDivisionError:
                    reward_greed = 0.0

                reward = [r * (reward_sample-reward_greed) for r in reward]

                # if reward_greed < reward_sample:
                #     sample_sent = [self.data.vocab_simple.describe(w) for w in cur_sample_target_list]
                #     greed_sent = [self.data.vocab_simple.describe(w) for w in cur_greed_target_list]
                #     simp_sent = [self.data.vocab_simple.describe(w) for w in cur_gt_simp_list]
                #     comp_sent = [self.data.vocab_complex.describe(w) for w in cur_gt_comp_list]
                #     print('sample_sent:%s\ngreed_sent:%s\nsimp_sent:%s\ncomp_sent:%s\nreward:%s\n=====\n' %
                #           (sample_sent, greed_sent, simp_sent, comp_sent, reward))


            # if 'rule' in self.model_config.rl_configs:
            #     rule_weight = self.model_config.rl_configs['rule_weight']
            #     reward_sample = [1.0 for _ in range(num_steps)]
            #     reward_greed = [1.0 for _ in range(num_steps)]
            #     cur_rule_target_input_placeholder = rule_target_input_placeholder[batch_i]
            #     for step in range(num_steps):
            #         if cur_sample_target_list[step] in cur_rule_target_input_placeholder:
            #             reward_sample[step] *= rule_weight
            #         if cur_greed_target_list[step] in cur_rule_target_input_placeholder:
            #             reward_greed[step] *= rule_weight
            #     reward = [r * (reward_sample[step] - reward_greed[step]) for step, r in enumerate(reward)]

            rewards.append(reward)
        return np.array(rewards, dtype=np.float32)

if __name__ == '__main__':
    cur_sample_target_list = [1,2,30,4,50]
    cur_greed_target_list = [1,2,30,4,5]
    cur_gt_simp_list = [1,2,30,4,50]
    cur_gt_comp_list = [1,2,3,4,5]

    reward = [1.0 for _ in range(5)]

    cur_sample_target_str = ' '.join([str(o) for o in cur_sample_target_list])
    cur_greed_target_str = ' '.join([str(o) for o in cur_greed_target_list])
    cur_gt_simp_str = ' '.join([str(o) for o in cur_gt_simp_list])
    cur_gt_comp_str = ' '.join([str(o) for o in cur_gt_comp_list])
    try:
        reward_sample = 1 * SARIsent(cur_gt_comp_str, cur_sample_target_str, [cur_gt_simp_str]) + (
                    1 - 1) * SARIsent(cur_gt_comp_str, cur_gt_simp_str, [cur_sample_target_str])
    except ZeroDivisionError:
        reward_sample = 0.0
    try:
        reward_greed = 1 * SARIsent(cur_gt_comp_str, cur_greed_target_str, [cur_gt_simp_str]) + (
                    1 - 1) * SARIsent(cur_gt_comp_str, cur_gt_simp_str, [cur_greed_target_str])
    except ZeroDivisionError:
        reward_greed = 0.0
    reward = [r * (reward_sample-reward_greed) for r in reward]
    print(reward)



