import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
#from mne.time_frequency import psd_welch
#https://mne.tools/1.4/auto_tutorials/time-freq/10_spectrum_class.html
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

import os

## logging info ###
import logging
from datetime import datetime

import os
os.environ["QT_API"] = "pyside6"

# logname = './logs/'+ datetime.now().strftime('log_%Y-%m-%d.log')
# logging.basicConfig(filename=logname,level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# logger = logging.getLogger(__name__)

fmin = 0
fmax = 45

freq_bands = {"delta": [0.5, 4.0],
                "theta": [4.0, 8.0],
                "alpha": [8.0, 13.0],
                "beta": [13.0, 30.0],
                "sigma": [30.0, 40.0],}

# Compute the power spectrum of epochs from 'epoch_connectivity' (not linked to ERP)

def epo_spectrum_overSubs(subs, selected_chans, proto, cfg, save=True, plot=True, show_plot=False):
    
    data = []
    
    for i_sub, sub in enumerate(subs):
        fif_path = os.path.join(cfg.data_epochs_path, f"{sub}_{proto}{cfg.prefix_epo_conn}")
        print('fif_fname : ', fif_path)
        epochs = mne.read_epochs(fif_path, proj=False, verbose=True, preload=True)  

        if sub in bad_subs:
            epochs.info['bads'] = []
        
        #Compute epochs spectrums
        epo_spectrum = epochs.compute_psd(fmin=fmin, fmax=fmax, picks=selected_chans)
        
        # compute evoked to be hacked with averaged spectrum accross epocks and subjects 
        evoked = epochs.average()
        evk_spectrum = evoked.compute_psd(fmin=fmin, fmax=fmax, picks = selected_chans)
        
        # show plots and save if needed
        if plot:
             #create plots
            epo_fft_fig = epo_spectrum.plot(show=False)
            sub_title = fif_path.split('/')[-1].replace('.fif', ' spectrum')
            epo_fft_fig.suptitle(f'{sub_title}')   
        
            epo_fft_topo_fig = epo_spectrum.plot_topo(show=False)
            epo_fft_topo_fig.suptitle(f'{sub_title} topoplot') 
            
            if save:
                fname_sub_fig = os.path.join(cfg.data_save_dir, f"{sub}_{proto}_epo_spectrum.png")
                fname_topo_fig = os.path.join(cfg.data_save_dir, f"{sub}_{proto}_topoplot.png")
                
                # Save the figures
                epo_fft_fig.savefig(fname_sub_fig)
                epo_fft_topo_fig.savefig(fname_topo_fig)

            plt.close(epo_fft_fig)
            plt.close(epo_fft_topo_fig)

        #get data to compute gran average
        psds, freqs = epo_spectrum.get_data(return_freqs=True)  # shape: (n_epochs, n_channels, n_freqs)
        print(f"\nPSDs shape: {psds.shape}, freqs shape: {freqs.shape}")
        print('freqs : ', freqs)
        print(type(freqs))
        print('indexs : ', np.where(np.logical_and(freqs>=6, freqs<=10)))
        avg_psd = np.average(psds, axis=0)  # Compute average PSD; shape: (n_channels, n_freqs)
        print(f"{sub}: avg_psd.shape = {avg_psd.shape}")  # Debug: print the shape for the current subject
        data.append(avg_psd)  # Append the average PSD to the list
        # shape psd : (n_epochs, n_chan, n_freqs)
        
        idx_epo = range(psds.shape[0])
        sub_df_band = pd.DataFrame(np.nan, index=idx_epo, columns=freq_bands.keys())
        for bb in freq_bands.keys():
            sub_idx_low = np.where(freqs >= freq_bands[bb][0])[0][0]
            sub_idx_high = np.where(freqs <= freq_bands[bb][1])[0][-1]
            sub_bb_average = np.average(psds[:, :, sub_idx_low:sub_idx_high], axis=(1, 2)) # shape (n_epochs, n_chan, n_freqs)
            sub_df_band.loc[:, bb] = sub_bb_average
        # Ajout des puissances relatives
        total_power = sub_df_band.sum(axis=1)  # Somme sur toutes les bandes pour chaque epoch
        for bb in freq_bands.keys():
            sub_df_band[f"{bb}_relative"] = sub_df_band[bb] / total_power
        # export excel
        df_sub_band_name = os.path.join(cfg.data_save_dir, f"{sub}_{proto}_freq_band_av.xlsx")
        sub_df_band.to_excel(df_sub_band_name)
        
    np_data = np.asarray(data)
    #print(np_data.shape)
    av_spectrum = np.average(np_data, axis=0) # shape np_data : (n_subs, n_chan, n_freqs)

    #hack evocked spectrum to plot the grand average
    #print(evk_spectrum._data.shape)
    evk_spectrum._data = av_spectrum
    
    av_spectrum_fig = evk_spectrum.plot(show=False)
    av_spectrum_fig_title = f'Averaged spectrum over subjects for {proto} protocol'
    av_spectrum_fig.suptitle(av_spectrum_fig_title) 
    fname_av_spectrum = os.path.join(cfg.data_save_dir, f"{proto}_averaged_epo_psd.png")
    av_spectrum_fig.savefig(fname_av_spectrum)
    plt.close(av_spectrum_fig)
    
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
    df_band_name = os.path.join(cfg.data_save_dir, f"{proto}_freq_band_av.xlsx")
    df_band.to_excel(df_band_name)
        
        
    return av_spectrum, freqs

if __name__ == '__main__':
    
    save = True
    verbose = False
    plot = True
    show_plot = False

    # Set the save directory as desired
    # Example: data_save_dir = r"C:\Users\adminlocal\Desktop\ConnectDoc\EEG_2025_CAP_FPerrin_Vera\Analysis_Baking_EEG_Vera\spectral_power\psd_del+"
    # (Make sure cfg.data_save_dir is updated accordingly in the Config class below.)
    
    class Config:
        data_save_dir = r"C:\Users\adminlocal\Desktop\ConnectDoc\EEG_2025_CAP_FPerrin_Vera\Analysis_Baking_EEG_Vera\spectral_power\psd_del-"
        data_epochs_path = r"C:\Users\adminlocal\Desktop\ConnectDoc\EEG_2025_CAP_FPerrin_Vera\Analysis_Baking_EEG_Vera\data_connectivity"
        prefix_epo_conn = '_epo_conn.fif'
    
    cfg = Config()
    
    common_channels =  ['E62']
    


    bad_subs = []
    subs = ["TpAC23", "TpBD16", "TpCG36", "TpFF34", "TpGB8", "TpJA20", "TpJPG7", "TpJPL10",
        "TpLP11", "TpMD13", "TpME22", "TpPA35", "TpSD30", "TpYB41"]
    '''
     subs =  [
    "TpAK24",
    "TpAK27",
    "TpCB15",
    "TpDRL3",
    "TpJB25",
    "TpJLR17",
    "TpMB45",
    "TpMN42",
    "TpPC21",
    "TpPM14",
    "TpRD38"
    ]

    '''

    #proto = ['LG','PP','Resting']
    proto = ['Resting']
    selected_chans = 'all' #common_channels
    
    # Loop over protocols (preserving the original loop)
    for p in proto:
        print(f"\n=== Processing protocol: {p} ===\n")
        epo_spectrum_overSubs(subs, selected_chans, p, cfg, save=save, plot=plot, show_plot=show_plot)


    
    ###################### On one subject #####################################
    
    '''
    sub = 'GT50'
    proto = 'PP' 
    fif_fname = f'{cfg.data_epochs_path}{sub}_{proto}{cfg.prefix_epoched}'
    selected_chans = 'all'#['E32', 'E25', 'E26']

    con_data = epo_spectrum_1sub(fif_fname, selected_chans)
   
    '''
    
    
