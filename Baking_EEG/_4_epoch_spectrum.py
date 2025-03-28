
import matplotlib
import mne
#from mne.time_frequency import psd_welch
#https://mne.tools/1.4/auto_tutorials/time-freq/10_spectrum_class.html
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

import os

## logging info ###
import logging
from datetime import datetime

import os
os.environ["QT_API"] = "pyside6"

logname = './logs/'+ datetime.now().strftime('log_%Y-%m-%d.log')
logging.basicConfig(filename=logname,level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


fmin = 0
fmax = 45


freq_bands = {"delta": [0.5, 4.0],
                "theta": [4.0, 8.0],
                "alpha": [8.0, 13.0],
                "beta": [13.0, 30.0],
                "sigma": [30.0, 40.0],}

# Compute the power spectrum of epochs from 'epoch_connectivity' (not linked to ERP)

def epo_spectrum_overSubs(subs, selected_chans, proto, cfg, save=True, plot=True, show_plot=True):
    
    data = []
    
    for i_sub, sub in enumerate(subs):
        fif_fname = f'{cfg.data_save_dir}{cfg.data_epochs_path}{sub}_{proto}{cfg.prefix_epo_conn}'
        print('fif_fname : ', fif_fname)
        epochs = mne.read_epochs(fif_fname, proj=False, verbose=True, preload=True)  
        
        #Compute epochs spectrums
        epo_spectrum = epochs.compute_psd(fmin=fmin, fmax=fmax, picks=selected_chans)
        
        # compute evoked to be hacked with averaged spectrum accross epocks and subjects 
        evoked = epochs.average()
        evk_spectrum = evoked.compute_psd(fmin=fmin, fmax=fmax, picks = selected_chans)
        
        #show plots and save if needed
        if plot:
             #create plots
            epo_fft_fig = epo_spectrum.plot()
            sub_title = fif_fname.split('/')[-1].replace('.fif', '')
            epo_fft_fig.suptitle(f'{sub_title} spectrum')   
        
            epo_fft_topo_fig = epo_spectrum.plot_topo()
            epo_fft_topo_fig.suptitle(f'{sub_title} spectrum topoplot') 
        
            plt.show()
        
            if save:
                if not os.path.exists(f'{cfg.data_save_dir}{cfg.data_psd_path}/{sub}/'):
                    os.makedirs(f'{cfg.data_save_dir}{cfg.data_psd_path}/{sub}/')
                
                fname_sub_fig =  f'{cfg.data_save_dir}{cfg.data_psd_path}/{sub}/{sub}_{proto}_epo_spectrum.png'
                epo_fft_fig.savefig(fname_sub_fig)

        #get data to compute gran average
        psds, freqs = epo_spectrum.get_data(return_freqs=True)
        print(f"\nPSDs shape: {psds.shape}, freqs shape: {freqs.shape}")
        print('freqs : ', freqs)
        print(type(freqs))
        print('indexs : ', np.where(np.logical_and(freqs>=6, freqs<=10)))
        data.append(np.average(psds, axis=0))  # shape psd : (n_epochs, n_chan, n_freqs)
        
        idx_epo = range(psds.shape[0])
        sub_df_band = pd.DataFrame(np.nan, index=idx_epo, columns=freq_bands.keys())
        for bb in freq_bands.keys():
            sub_idx_low = np.where(freqs >= freq_bands[bb][0])[0][0]
            sub_idx_high = np.where(freqs <= freq_bands[bb][1])[0][-1]
            sub_bb_average = np.average(psds[:, :, sub_idx_low:sub_idx_high], axis=(1, 2)) # shape (n_epochs, n_chan, n_freqs)
            #print('shape bb_average ', bb_average.shape)
            sub_df_band.loc[:, bb] = sub_bb_average
        # Ajout des puissances relatives
        total_power = sub_df_band.sum(axis=1)  # Somme sur toutes les bandes pour chaque epoch
        for bb in freq_bands.keys():
            sub_df_band[f"{bb}_relative"] = sub_df_band[bb] / total_power
        # export excel
        df_sub_band_name = f'{cfg.data_save_dir}{cfg.data_psd_path}/{sub}/{sub}_{proto}_freq_band_av.xlsx'
        sub_df_band.to_excel(df_sub_band_name)
        
    np_data = np.asarray(data)
    #print(np_data.shape)
    av_spectrum = np.average(np_data, axis=0) # shape np_data : (n_subs, n_chan, n_freqs)

    #hack evocked spectrum to plot the grand average
    #print(evk_spectrum._data.shape)
    evk_spectrum._data = av_spectrum
    
    av_spectrum_fig = evk_spectrum.plot()
    av_spectrum_fig_title = f'Averaged spectrum over subjects for {proto} protocol'
    av_spectrum_fig.suptitle(av_spectrum_fig_title) 
    fname_av_spectrum =  f'{cfg.data_save_dir}{cfg.data_psd_path}/{proto}_averaged_epo_psd.png'
    av_spectrum_fig.savefig(fname_av_spectrum)
    plt.show()
    
    #Export average power by band over chan by subject
    df_band = pd.DataFrame(np.nan, index=subs, columns=freq_bands.keys())
    
    for bb in freq_bands.keys():
        
        idx_low = np.where(freqs >= freq_bands[bb][0])[0][0]
        idx_high = np.where(freqs <= freq_bands[bb][1])[0][-1]
        #print(f'idx_low: {idx_low} and idx_high: {idx_high}')
        #print('shape np_data ', np_data.shape)
        #print(f' shape de {np_data[:, :, idx_low:idx_high].shape}')
        bb_average = np.average(np_data[:, :, idx_low:idx_high], axis=(1, 2)) # shape (n_subs, n_chan, n_freqs)
        #print('shape bb_average ', bb_average.shape)
        df_band.loc[:, bb] = bb_average
        
    total_power_all = df_band.sum(axis=1)  # Somme sur toutes les bandes pour chaque epoch
    for bb in freq_bands.keys():
        df_band[f"{bb}_relative"] = df_band[bb] / total_power_all
    #print(df_band)
    df_band_name =  f'{cfg.data_save_dir}{cfg.data_psd_path}/{proto}_freq_band_av.xlsx'
    df_band.to_excel(df_band_name)
        
        
        
    

        
                
def epo_spectrum_1sub(fif_fname, selected_chans, plot=True):

    #load epochs
    epochs = mne.read_epochs(fif_fname, proj=False, verbose=True, preload=True).pick_types(eeg=True) 
    #epochs = epochs[event_id].pick_channels(selected_chans, ordered=True)

    # Epochs spectrum computation
    epo_spectrum = epochs.compute_psd(picks = selected_chans)
   
    #Plots
    epo_fft_fig = epo_spectrum.plot()
    sub_title = fif_fname.split('/')[-1].replace('.fif', '')
    epo_fft_fig.suptitle(f'{sub_title} spectrum')   
    
    epo_fft_topo_fig = epo_spectrum.plot_topo()
    epo_fft_topo_fig.suptitle(f'{sub_title} spectrum topoplot')
    
    plt.show()
   

    
if __name__ == '__main__':
    
    #'''
    save = False
    verbose = False
    plot = False
    show_plot = False
    
    subs = ['CS38', 'CS38']  #AP84
    proto = 'PP'  #'LG' 'Rest'
    selected_chans = 'all'  #['E32', 'E25', 'E26']
    
    epo_spectrum_overSubs(subs, selected_chans, proto, save, plot, show_plot)
    #'''
    
    
    ###################### On one subject #####################################
    
    '''
    sub = 'GT50'
    proto = 'PP' 
    fif_fname = f'{cfg.data_epochs_path}{sub}_{proto}{cfg.prefix_epoched}'
    selected_chans = 'all'#['E32', 'E25', 'E26']

    con_data = epo_spectrum_1sub(fif_fname, selected_chans)
   
    '''
    
    
