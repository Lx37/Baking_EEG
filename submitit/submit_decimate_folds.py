
#!/usr/bin/env python3
"""
Script de soumission pour les tests de décimation et folds sur cluster SLURM.
Version adaptée sans feature selection.
"""

import os
import sys
import logging
from datetime import datetime

# Ajouter le chemin vers les modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import submitit
except ImportError:
    print("ERROR: submitit n'est pas installé. Installez avec: pip install submitit")
    sys.exit(1)

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PARAMÈTRES À MODIFIER ICI
SUBJECT_FILE = "/mnt/data/tom.balay/data/Baking_EEG_data/PP_COMA_01HZ/Battery/YG72_preproc_ICA_PPAP-epo_ar.fif"

QUICK_TEST = False  # Mettre False pour analyse complète

# Configuration cluster
PARTITION = "CPU"
MEM = "60G"
CPUS = 40
TIMEOUT_MIN = 7200  # 12 heures en minutes
ACCOUNT = "tom.balay"

def create_job_function():
    """
    Crée la fonction qui sera exécutée sur le cluster.
    """
    def job_function():
        import sys
        import os
        from datetime import datetime
        
        # Ajouter le chemin vers les modules
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        sys.path.insert(0, project_root)
        
        print(f"=== DÉBUT DU JOB SUR LE CLUSTER ===")
        print(f"Timestamp: {datetime.now()}")
        print(f"Fichier sujet: {SUBJECT_FILE}")
        print(f"Mode rapide: {QUICK_TEST}")
        print(f"Working directory: {os.getcwd()}")
        print(f"Python path: {sys.path[:3]}")
        
        try:
            # Import de la fonction de test après avoir configuré le path
            from examples.test_decimate_folds import run_quick_test, run_comprehensive_analysis, save_results_and_visualizations
            
            # Modifier le fichier sujet global
            import examples.test_decimate_folds as test_module
            test_module.TEST_SUBJECT_FILE = SUBJECT_FILE
            
            if QUICK_TEST:
                print("Lancement du test rapide...")
                output_dir = run_quick_test()
            else:
                print("Lancement de l'analyse complète...")
                results_df, subject_info = run_comprehensive_analysis()
                output_dir = save_results_and_visualizations(results_df, subject_info)
            
            print(f"Analyse terminée avec succès!")
            print(f"Résultats sauvegardés dans: {output_dir}")
            return output_dir
                
        except Exception as e:
            print(f"ERREUR durant l'exécution: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print(f"=== FIN DU JOB SUR LE CLUSTER ===")
    
    return job_function

def main():
    """
    Fonction principale pour soumettre le job.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Soumission des tests de décimation sur cluster')
    parser.add_argument('--quick', action='store_true', 
                       help='Exécuter un test rapide avec paramètres réduits')
    parser.add_argument('--local', action='store_true', 
                       help='Exécuter localement au lieu de soumettre sur cluster')
    parser.add_argument('--subject-file', type=str, default=None,
                       help='Fichier de données du sujet (optionnel)')
    
    args = parser.parse_args()
    
    # Modifier les paramètres selon les arguments
    global QUICK_TEST, SUBJECT_FILE
    if args.quick:
        QUICK_TEST = True
    if args.subject_file:
        SUBJECT_FILE = args.subject_file
    
    logger.info(f"Configuration:")
    logger.info(f"  - Mode rapide: {QUICK_TEST}")
    logger.info(f"  - Fichier sujet: {SUBJECT_FILE}")
    
    if args.local:
        logger.info("Exécution locale...")
        job_function = create_job_function()
        output_dir = job_function()
        logger.info(f"Terminé! Résultats dans: {output_dir}")
        return
    
    # Soumission sur cluster
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_dir = f"logs_decimate_folds_{timestamp}"
    
    logger.info(f"Soumission du job sur le cluster...")
    logger.info(f"Logs seront sauvegardés dans: {logs_dir}")
    
    # Configuration pour l'environnement
    setup_commands = [
        "module load python/3.11",
        "source ~/.venvs/py3.11_cluster/bin/activate",
    ]
    
    try:
        # Configuration de l'executor
        executor = submitit.AutoExecutor(folder=logs_dir)
        executor.update_parameters(
            timeout_min=TIMEOUT_MIN,
            slurm_partition=PARTITION,
            slurm_mem=MEM,
            slurm_cpus_per_task=CPUS,
            slurm_additional_parameters={"account": ACCOUNT},
            slurm_job_name="decimate_folds_analysis",
            slurm_setup=setup_commands,
        )
        
        # Créer et soumettre le job
        job_function = create_job_function()
        job = executor.submit(job_function)
        
        logger.info(f"Job soumis avec l'ID: {job.job_id}")
        logger.info(f"Logs disponibles dans: {logs_dir}")
        logger.info(f"Statut du job: {job.state}")
        logger.info("Utilisez 'squeue -u tom.balay' pour voir le statut")
        
        return job
    
    except Exception as e:
        logger.error(f"ERREUR lors de la soumission: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
