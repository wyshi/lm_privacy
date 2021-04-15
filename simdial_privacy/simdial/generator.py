# -*- coding: utf-8 -*-
# author: Tiancheng Zhao

from simdial.agent.user import User
from simdial.agent.system import System
from simdial.channel import ActionChannel, WordChannel
from simdial.agent.nlg import SysNlg, UserNlg
from simdial.complexity import Complexity
from simdial.domain import Domain
import progressbar
import json
import numpy as np
import sys
import os
from copy import deepcopy
import re

class Generator(object):
    """
    The generator class used to generate synthetic slot-filling human-computer conversation in any domain. 
    The generator can be configured to generate data with varying complexity at: propositional, interaction and social 
    level. 
    
    The required input is a domain specification dictionary + a configuration dict.
    """

    @staticmethod
    def pack_msg(speaker, utt, **kwargs):
        resp = {k: v for k, v in kwargs.items()}
        resp["speaker"] = speaker
        resp["utt"] = utt
        return resp

    @staticmethod
    def pprint(dialogs, in_json, domain_spec, output_file=None):
        """
        Print the dailog to a file or STDOUT
        
        :param dialogs: a list of dialogs generated
        :param output_file: None if print to STDOUT. Otherwise write the file in the path
        """
        f = sys.stdout if output_file is None else open(output_file, "w")
        def np_encoder(object):
            if isinstance(object, np.generic):
                return object.item()

        if in_json:
            combo = {'dialogs': dialogs, 'meta': domain_spec.to_dict()}
            json.dump(combo, f, indent=2,default=np_encoder)
        else:
            for idx, d in enumerate(dialogs):
                f.write("## DIALOG %d ##\n" % idx)
                for turn in d:
                    speaker, utt, actions = turn["speaker"], turn["utt"], turn["actions"]
                    if utt:
                        str_actions = utt
                    else:
                        str_actions = " ".join([a.dump_string() for a in actions])
                    if speaker == "USR":
                        f.write("%s(%f)-> %s\n" % (speaker, turn['conf'], str_actions))
                    else:
                        f.write("%s -> %s\n" % (speaker, str_actions))

        if output_file is not None:
            f.close()

    @staticmethod
    def print_stats(dialogs):
        """
        Print some basic stats of the dialog.
        
        :param dialogs: A list of dialogs generated.
        """
        print("%d dialogs" % len(dialogs))
        all_lens = [len(d) for d in dialogs]
        print("Avg len {} Max Len {}".format(np.mean(all_lens), np.max(all_lens)))
        # We don't need to print knowledge base related stats because we removed KB for this task
        # total_cnt = 0.
        # kb_cnt = 0.
        # ratio = []
        # for d in dialogs:
        #     local_cnt = 0.
        #     for t in d:
        #         total_cnt +=1
        #         if 'QUERY' in t['utt']:
        #             kb_cnt += 1
        #             local_cnt += 1
        #     ratio.append(local_cnt/len(d))
        # print(kb_cnt/total_cnt)
        # print(np.mean(ratio))

    def gen(self, domain, complexity, num_sess=1, special_domain=None):
        """
        Generate synthetic dialogs in the given domain. 

        :param domain: a domain specification dictionary
        :param complexity: an implmenetaiton of Complexity
        :param num_sess: how dialogs to generate
        :return: a list of dialogs. Each dialog is a list of turns.
        """
        dialogs = []

        domain = Domain(special_domain)
        bar = progressbar.ProgressBar(max_value=num_sess)

        for i in range(num_sess):
            bar.update(i)
            domain_spec_copy = deepcopy(special_domain)

            # for Track pacakge domain, we randomize question order each time
            if domain.name == "track_package":
                
                num_info_ask = special_domain.num_info_ask
                #print("NUM INFO ASK:",num_info_ask)
                all_user_slots = [slot_name for (slot_name,_,_) in special_domain.usr_slots]
                flex_user_slots = all_user_slots[1:]
                chosen_slots = ["name"] + list(np.random.choice(flex_user_slots, num_info_ask))
                #print("CHOSEN USER SLOTS:", chosen_slots)
                unwanted = set(all_user_slots) - set(chosen_slots)
                remain_user_slots = [(slot_name,des,opt) for (slot_name,des,opt) in special_domain.usr_slots if slot_name in chosen_slots]
                
                for unwanted_key in unwanted: 
                    del domain_spec_copy.nlg_spec[unwanted_key]
                domain_spec_copy.usr_slots = remain_user_slots
                domain = Domain(domain_spec_copy)

            action_channel = ActionChannel(domain, complexity)
            word_channel = WordChannel(domain, complexity)

            # natural language generators
            sys_nlg = SysNlg(domain, complexity)
            usr_nlg = UserNlg(domain, complexity)

            usr = User(domain, complexity)
            sys = System(domain, complexity)

            # begin conversation
            noisy_usr_as = []
            dialog = []
            conf = 1.0
            while True:
                # make a decision
                sys_r, sys_t, sys_as, sys_s = sys.step(noisy_usr_as, conf)
                #print("SYSTEM ACTIONS", sys_as)
                sys_utt, sys_str_as = sys_nlg.generate_sent(sys_as, domain=domain)
                if sys_utt != "":
                    dialog.append(self.pack_msg("SYS", sys_utt, actions=sys_str_as, domain=domain.name, state=sys_s))

                if sys_t:
                    break

                usr_r, usr_t, usr_as = usr.step(sys_as)

                # passing through noise, nlg and noise!
                noisy_usr_as, conf = action_channel.transmit2sys(usr_as)
                usr_utt = usr_nlg.generate_sent(noisy_usr_as)
                noisy_usr_utt = word_channel.transmit2sys(usr_utt)
                if noisy_usr_utt != "":
                    dialog.append(self.pack_msg("USR", noisy_usr_utt, actions=noisy_usr_as, conf=conf, domain=domain.name))

            dialogs.append(dialog)

        return dialogs

    def gen_corpus(self, name, domain_spec, complexity_spec, size):
        if not os.path.exists(name):
            os.mkdir(name)

        # create meta specifications
        domain = Domain(domain_spec)
        complex = Complexity(complexity_spec)

        # generate the corpus conditioned on domain & complexity
        corpus = self.gen(domain, complex, num_sess=size, special_domain=domain_spec)

        # txt_file = "{}-{}-{}.{}".format(domain_spec.name,
        #                                complexity_spec.__name__,
        #                                size, 'txt')

        json_file = "{}-{}-{}.{}".format(domain_spec.name,
                                         complexity_spec.__name__,
                                         size, 'json')

        json_file = os.path.join(name, json_file)
        self.pprint(corpus, True, domain_spec, json_file)
        self.print_stats(corpus)
