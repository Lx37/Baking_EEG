import mne
import numpy as np
import matplotlib.pyplot as plt
import os


#import config_protoPPmusic_ConvolutionEGI as cfg
## logging info ###
import logging
from datetime import datetime

logname = './logs/' + datetime.now().strftime('log_%Y-%m-%d.log')
logging.basicConfig(filename=logname,level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)



def create_evoked(data_epoch, patient_info, cfg, save=True, verbose=True, plot=True):
#create_ERP_Gtec(epochs, sub_name, cfg, fif_fname, protocol, plot=True, save=True):

    '''
    This function create evoked data and plots
    It supposes that raw data have been preprocessed and epoked data is saved to a .fif file
    (for exemple in data_preproc & data_erp folder)
    Averaged evoked data is saved in 'data_evoked'
    Figures if created are saved in the plots folders

    '''
    #fif_base_name = epochs_name.split('/')[-1].strip('.fif') + '_'
    fif_base_name = patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_epochs_PPAP.strip('.fif')

    print('ici : ', list(cfg.all_conditions_PP.values()))

    condEvokeds = [data_epoch[con].average() for con in  list(cfg.all_conditions_PP.values())]# ["PP","AP/1","AP/2","AP/3","AP/4","AP/5","AP/6"]] #TODO instead of this link to all_conditions?
    print(condEvokeds)
    for index, con in enumerate(list(cfg.all_conditions_PP)):
        print('index : ',index)
        print('con : ', con)
        condEvokeds[index].comment = con
        '''
        if plot:

            condEvokedFig = condEvokeds[index].plot(spatial_colors=True, titles=title, gfp=True, show=show, exclude='bads') #ylim=dict(eeg=[-10,8]),
            if save:
                png_name = patient_info['data_save_dir'] + cfg.all_folders_PP['plot_erp_path'] +  fif_base_name + con +'.png'
                condEvokedFig.savefig(png_name)
        if topo:
            condTopoFig = condEvokeds[index].plot_topomap(times= np.arange(0,1.2,0.1), ch_type='eeg', show=show, title=title)
            if save:
                png_name = patient_info['data_save_dir'] + cfg.all_folders_PP['plot_topo_path'] +  fif_base_name + con +'.png'
                #print(png_name)
                condTopoFig.savefig(png_name)
        '''

    evoked_dict = dict()

    for con in cfg.all_conditions_PP:

        print('################"condition : ', con)
        logger.info("condition: ,%s", con)
        logger.info("nb events: ,%s", len(data_epoch[cfg.all_conditions_PP[con]]))
        print("nb events:", len(data_epoch[cfg.all_conditions_PP[con]]))

        condEvoked = data_epoch[cfg.all_conditions_PP[con]].average()
        condEvoked.comment = con
        title = patient_info['ID_patient']  + ' ' + con
        condEvokedFig = condEvoked.plot(spatial_colors=True, titles=title, gfp=True, show=plot, exclude='bads') #ylim=dict(eeg=[-10,8]),
        if save:
            png_name = patient_info['data_save_dir'] + cfg.all_folders_PP['plot_erp_path'] +  fif_base_name + '_' +  con +'.png'
            condEvokedFig.savefig(png_name)
        if cfg.erp_topo:
            condTopoFig = condEvoked.plot_topomap(times= np.arange(0,1,0.1), ch_type='eeg', show=plot)
            plt.title(title)
            if save:
                png_name = patient_info['data_save_dir'] + cfg.all_folders_PP['plot_topo_path'] +  fif_base_name + '_' +  con +'.png'
                #print(png_name)
                condTopoFig.savefig(png_name)



    if save:
        #AllConditions_name = cfg.data_evoked_path + fif_base_name + cfg.prefix_ave
        AllConditions_name = patient_info['data_save_dir'] + cfg.all_folders_PP['data_evoked_path'] + fif_base_name + cfg.prefix_ave
        print(AllConditions_name)
        mne.write_evokeds(AllConditions_name, condEvokeds, overwrite=True)
        logger.info('Saved evoked (ERP) datafile', AllConditions_name)

    return condEvokeds

