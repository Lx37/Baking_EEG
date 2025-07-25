import mne
import numpy as np
import matplotlib.pyplot as plt
import os

from mne.preprocessing import ICA
# from autoreject import get_rejection_threshold
# from autoreject import (AutoReject, set_matplotlib_defaults)


# ## logging info ###
# import logging
# from datetime import datetime

# #logname = '/home/lizette/Projets/GIT_LAB/coma_stuff/logs/' + datetime.now().strftime('log_%Y-%m-%d.log')
# logname = './logs/' + datetime.now().strftime('log_%Y-%m-%d.log')
# logging.basicConfig(filename=logname,level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# logger = logging.getLogger(__name__)


def correct_blink_ICA(data, patient_info, cfg, save=False, verbose=True, plot=True):
        
    data.plot(block=True, title='Show preprocessed EEG to see if there is ocular events')

    eog_chan = 'VEOGL'
    
    eog_epoch = mne.preprocessing.create_eog_epochs(data, ch_name=eog_chan)
    eog_evoked = eog_epoch.average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))
    n_blinks = eog_evoked.nave
    
    if verbose:
        print(f"Number of blink found: {n_blinks}")

    if plot:
        data.plot(events=eog_epoch.events, title='Show EOG artifacts detection')
        eog_evoked.plot_joint(title='Show averaged EOG artifacts')

    # logger.info('Nb blink found : ,%s', n_blinks)
    
    
    ################ ICA correction ##################
    if n_blinks >cfg.minBlinksICA:
    
        filt_raw = data.copy().filter(l_freq=1.0, h_freq=None)
    
        ica = ICA(n_components=cfg.n_components, max_iter="auto", random_state=97)
        ica.fit(filt_raw)
    
        explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
        for channel_type, ratio in explained_var_ratio.items():
            print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

        ica.plot_components()
    
        # blinks
        #ica.plot_overlay(data, exclude=[0,6], picks="eeg")
        ica.plot_overlay(data, exclude=np.array([0,6]), picks="eeg", title='Plot Overlay')
        #ica.plot_properties(data, picks=slice(0,15,1))
    
        # find which ICs match the EOG pattern
        eog_comp_indices, eog_scores = ica.find_bads_eog(data, ch_name=eog_chan)
        ica.exclude = eog_comp_indices
        
        print('ICA eog_comp_indices : ', eog_comp_indices)
        print('ica.exclude : ', ica.exclude)

        # barplot of ICA component "EOG match" scores
        ica.plot_scores(eog_scores)

        # plot ICs applied to raw data, with EOG matches highlighted
        # possibility in this plot to select/unselect ICs
        ica.plot_sources(data, show_scrollbars=False, block=True, title="Sources over time. Please select / unselect the ones you want to remove")

        # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
        ica.plot_sources(eog_evoked)
        
        # plot diagnostics
        #if ica.exclude != []:
        #    ica.plot_properties(data)
        
        print('ICA eog_comp_indices : ', eog_comp_indices)
        print('ica.exclude : ', ica.exclude)
        
        if plot:
            data.plot(events=eog_epoch.events, title="EEG signal before ICA correction")
        
        #Apply ICA
        #ica.exclude.extend(eog_comp_indices)
        ica.apply(data)#, exclude=eog_comp_indices)
        # logger.info('components excluded = %s',  ica.exclude)
        
        if plot:
            data.plot(events=eog_epoch.events, block=True, title="EEG signal after ICA correction")
        
        if save and ica.exclude!= []:
            data_name = patient_info['data_save_dir'] + cfg.data_preproc_path
            data_name = data_name + patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_ICA

            #data_name = cfg.data_preproc_path + data.info['subject_info']['his_id'] + cfg.prefix_processed.strip('.fif') + cfg.prefix_ICA
            data.save(data_name, overwrite=True)
            # logger.info('saved ICA data', data_name)
            return data
        else:
            print('ICA not saved as no components were excluded')
    else:
        print('Not enough blinks found to perform ICA')
        exit()
        
    '''
    if save:
        data_name = patient_info['data_save_dir'] + cfg.all_folders_PP['data_preproc_path']
        data_name = data_name + patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_noICA

        #data_name = cfg.data_preproc_path + data.info['subject_info']['his_id'] + cfg.prefix_processed.strip('.fif') + cfg.prefix_noICA
        data.save(data_name, overwrite=True)
        logger.info('saved ICA data', data_name)
    '''

    return data
    
def autorejection_epochs(cfg, epochs,fif_fname, protocol, save=False, verbose=True, plot=True):

    n_interpolates = np.array([1, 4, 32])
    consensus_percs = np.linspace(0, 1.0, 11)
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, stim=False,eog=False, exclude=[])

    ar = AutoReject(n_interpolate=n_interpolates, consensus=consensus_percs, picks=picks,
                thresh_method='random_search', random_state=42)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)


    if plot:
        reject_log.plot_epochs(epochs)
        ar.get_reject_log(epochs).plot()
        epochs_clean.plot(title='Epochs after cleaning')
        epochs_clean.plot_drop_log()

    old = epochs.average()
    png_name = cfg.cleaning_path +  fif_fname.split('/')[-1].strip('.fif') + '_old.png'

    old.plot(spatial_colors=True, show=False, titles='Old data').savefig(png_name)

    new = epochs_clean.average()
    png_name = cfg.cleaning_path +  fif_fname.split('/')[-1].strip('.fif') + '_new.png'
    new.plot(spatial_colors=True, show=False, titles='Cleaned data').savefig(png_name)
    png_name = cfg.cleaning_path +  fif_fname.split('/')[-1].strip('.fif') + '_difference.png'
    difference = mne.combine_evoked([old, - new], weights='equal').plot_joint(show=False, title='Difference raw & cleaned').savefig(png_name)

    if plot:
        plt.show()

    print('Nb epochs uncleaned, %s',len(epochs))
    print('Nb epochs cleaned, %s',len(epochs_clean))
    print('Nb epochs deleted, %s',(len(epochs)-len(epochs_clean)))


    #logger.info('The instances of _AutoReject for each channel type. %s', ar.local_reject)
    #logger.info('Percentage dropped epochs, %s', epochs_clean.drop_log_stats)
    # logger.info('Nb epochs uncleaned, %s',len(epochs))
    # logger.info('Nb epochs cleaned, %s',len(epochs_clean))
    # logger.info('Nb epochs deleted, %s',(len(epochs)-len(epochs_clean)))

    if save:
        data_name = fif_fname.strip('.fif') + cfg.prefix_autoreject
        epochs_clean.save(data_name, overwrite=True)
        # logger.info('saved cleaned data', data_name)

    return epochs

def ICA_0Artifact (epochs, n_components, all_conditions):
    epochs["PP"].average().plot(spatial_colors=True)
    epochsfilt = epochs.copy()
    #epochsfilt.plot_psd(fmin=0.5,fmax=40,tmin=-0.03, tmax=0.03)
    #epochsfilt.plot_psd(tmin=-0.03, tmax=0.03)

    print (epochsfilt)

    epochsfilt.filter(5., 30., n_jobs=2, fir_design='firwin')
    ################ ICA correction ##################
    method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
    decim = 3  # we need sufficient statistics, not all time points -> saves time
    random_state = 23


    ica = ICA(n_components=n_components, method=method, random_state=random_state)
    picks = mne.pick_types(epochsfilt.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
    reject = {'eeg': 100e-6}
    ica.fit(epochsfilt, picks=picks, decim=3, reject=reject)
    print(ica)
    ica.plot_components()

    oaverage_filt = epochsfilt.average()
    oaverage_filt.plot(titles = 'Average ERP (all conditions)')

    ica.plot_sources(oaverage_filt)
    ica.plot_sources(epochsfilt, show=True, block=True)
    ica.plot_properties(epochsfilt, picks=(ica.exclude), psd_args={'fmax': 35.})
    ica.plot_overlay(oaverage_filt, exclude=(ica.exclude), show=False)


    ica.apply(epochs, exclude=(ica.exclude))
    oaverage = epochs.average()
    oaverage.plot(titles = 'Average ERP (all conditions) after 0-artifact ICA')
    epochs["PP"].average().plot(spatial_colors=True)

    return epochs

    ## for eyes you use find_bads_eog which uses pearson correlation to find scores matching components.
    ## TRY!! Not sure how to adapt the scripts in MNE to work for us.
    #o_inds, scores = ica.find_bads_eog(oaverage)
    #print(o_inds)
    #print(scores)
    #ica.plot_scores(scores, exclude=o_inds)
    #ica.plot_sources(eog_average, exclude=o_inds)
