#!/usr/bin/env python3
"""
Script submitit simplifié pour lancer les tests CSP/FS avec paramètres pré-définis.
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
SUBJECT_FILE = "/mnt/data/tom.balay/data/Baking_EEG_data/PP_PATIENTS_DELIRIUM+_0.5/TpSM49_PP_preproc_noICA_PP-epo_ar.fif"

QUICK_TEST = False  # Mettre False pour analyse complète

# Configuration cluster
PARTITION = "CPU"
MEM = "60G"
CPUS = 40
TIMEOUT_MIN = 12000 * 60  # en minutes
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
        print(f"Répertoire de sortie: {OUTPUT_DIR}")
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
            
            # Déplacer les résultats vers le répertoire de sortie final
            if OUTPUT_DIR and output_dir:
                import shutil
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                final_output = os.path.join(OUTPUT_DIR, f"cluster_results_{timestamp}")
                os.makedirs(os.path.dirname(final_output), exist_ok=True)
                shutil.move(output_dir, final_output)
                print(f"Résultats déplacés vers: {final_output}")
                return final_output
            else:
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
    
    # Vérifier que le fichier sujet existe
    if not os.path.exists(SUBJECT_FILE):
        logger.error(f"Fichier sujet non trouvé: {SUBJECT_FILE}")
        sys.exit(1)
    
    # Créer le répertoire de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Créer le répertoire de logs submitit
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_dir = os.path.join(os.path.dirname(__file__), f"logs_test_csp_fs_{timestamp}")
    os.makedirs(logs_dir, exist_ok=True)
    
    logger.info("=== CONFIGURATION DE SOUMISSION ===")
    logger.info(f"Fichier sujet: {SUBJECT_FILE}")
    logger.info(f"Répertoire de sortie: {OUTPUT_DIR}")
    logger.info(f"Mode rapide: {QUICK_TEST}")
    logger.info(f"Partition: {PARTITION}")
    logger.info(f"CPUs: {CPUS}")
    logger.info(f"Mémoire: {MEM}")
    logger.info(f"Timeout: {TIMEOUT_MIN} minutes")
    logger.info(f"Account: {ACCOUNT}")
    logger.info(f"Logs dans: {logs_dir}")
    logger.info("=" * 40)
    
    # Commandes de setup (modules à charger sur le cluster)
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
            local_setup=setup_commands,
        )
        
        # Créer et soumettre le job
        job_function = create_job_function()
        job = executor.submit(job_function)
        
        logger.info(f"Job soumis avec l'ID: {job.job_id}")
        logger.info(f"Logs disponibles dans: {logs_dir}")
        logger.info(f"Statut du job: {job.state}")
        
        # Optionnel : attendre le résultat
        # logger.info("Attente de la fin du job...")
        # result = job.result()  # Bloque jusqu'à la fin
        # logger.info(f"Job terminé. Résultats dans: {result}")
        
        return job
    
    except Exception as e:
        logger.error(f"ERREUR lors de la soumission: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
