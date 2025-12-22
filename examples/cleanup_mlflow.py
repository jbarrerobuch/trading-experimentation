"""
Utilidad para limpiar experimentos fallidos en MLflow
Elimina runs sin métricas, runs fallidos, o experimentos completos
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
from mlflow.tracking import MlflowClient


def list_all_experiments():
    """Lista todos los experimentos con estadísticas"""
    client = MlflowClient()
    experiments = client.search_experiments()
    
    print(f"\n{'='*100}")
    print(f"📊 EXPERIMENTOS EN MLFLOW")
    print(f"{'='*100}\n")
    
    for exp in experiments:
        if exp.name == "Default":
            continue
            
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=""
        )
        
        total_runs = len(runs)
        failed_runs = sum(1 for run in runs if run.info.status == 'FAILED')
        finished_runs = sum(1 for run in runs if run.info.status == 'FINISHED')
        running_runs = sum(1 for run in runs if run.info.status == 'RUNNING')
        
        # Runs sin métricas
        no_metrics = sum(1 for run in runs if not run.data.metrics)
        
        print(f"📁 {exp.name} (ID: {exp.experiment_id})")
        print(f"   Total runs: {total_runs}")
        print(f"   ✅ Finished: {finished_runs}")
        print(f"   ❌ Failed: {failed_runs}")
        print(f"   🔄 Running: {running_runs}")
        print(f"   ⚠️  Sin métricas: {no_metrics}")
        print()
    
    return experiments


def delete_failed_runs(experiment_name=None, dry_run=True):
    """
    Elimina runs fallidos o sin métricas
    
    Parameters:
    -----------
    experiment_name : str, optional
        Nombre del experimento específico. Si None, aplica a todos
    dry_run : bool
        Si True, solo muestra qué se eliminaría sin hacerlo
    """
    client = MlflowClient()
    
    # Obtener experimentos
    if experiment_name:
        try:
            experiment = client.get_experiment_by_name(experiment_name)
            experiments = [experiment] if experiment else []
        except:
            print(f"❌ Experimento '{experiment_name}' no encontrado")
            return
    else:
        experiments = client.search_experiments()
    
    total_deleted = 0
    
    for exp in experiments:
        if exp.name == "Default":
            continue
        
        print(f"\n📁 Procesando experimento: {exp.name}")
        
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=""
        )
        
        runs_to_delete = []
        
        for run in runs:
            # Criterios de eliminación
            is_failed = run.info.status == 'FAILED'
            no_metrics = not run.data.metrics
            is_running_stale = run.info.status == 'RUNNING'  # Runs que quedaron "running"
            
            if is_failed or no_metrics or is_running_stale:
                reason = []
                if is_failed:
                    reason.append("FAILED")
                if no_metrics:
                    reason.append("SIN_METRICAS")
                if is_running_stale:
                    reason.append("RUNNING_STALE")
                
                runs_to_delete.append((run.info.run_id, run.info.run_name, reason))
        
        if runs_to_delete:
            print(f"   Encontrados {len(runs_to_delete)} runs para eliminar:")
            
            for run_id, run_name, reasons in runs_to_delete:
                reason_str = ", ".join(reasons)
                print(f"   - {run_name} ({run_id[:8]}...) → {reason_str}")
                
                if not dry_run:
                    try:
                        client.delete_run(run_id)
                        total_deleted += 1
                    except Exception as e:
                        print(f"     ❌ Error eliminando: {e}")
        else:
            print(f"   ✅ No hay runs para eliminar")
    
    if dry_run:
        print(f"\n⚠️  DRY RUN - No se eliminó nada")
        print(f"   Se eliminarían {len(runs_to_delete)} runs")
        print(f"   Ejecuta con dry_run=False para eliminar")
    else:
        print(f"\n✅ Eliminados {total_deleted} runs")


def delete_experiment(experiment_name, dry_run=True):
    """
    Elimina un experimento completo con todos sus runs
    
    Parameters:
    -----------
    experiment_name : str
        Nombre del experimento a eliminar
    dry_run : bool
        Si True, solo muestra información sin eliminar
    """
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experimento '{experiment_name}' no encontrado")
            return
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=""
        )
        
        print(f"\n⚠️  ELIMINAR EXPERIMENTO: {experiment_name}")
        print(f"   ID: {experiment.experiment_id}")
        print(f"   Total runs: {len(runs)}")
        
        if dry_run:
            print(f"\n⚠️  DRY RUN - No se eliminó nada")
            print(f"   Ejecuta con dry_run=False para eliminar")
        else:
            confirm = input(f"\n❗ ¿Confirmar eliminación de {len(runs)} runs? (yes/no): ")
            if confirm.lower() == 'yes':
                # Eliminar todos los runs primero
                for run in runs:
                    try:
                        client.delete_run(run.info.run_id)
                    except Exception as e:
                        print(f"   ⚠️  Error eliminando run {run.info.run_id}: {e}")
                
                # Eliminar experimento
                client.delete_experiment(experiment.experiment_id)
                print(f"✅ Experimento '{experiment_name}' eliminado")
            else:
                print("❌ Cancelado")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def cleanup_old_sessions(experiment_name, keep_last_n=5, dry_run=True):
    """
    Mantiene solo las N últimas sesiones de un experimento
    Útil para experimentos que acumulan muchas sesiones de prueba
    
    Parameters:
    -----------
    experiment_name : str
        Nombre del experimento
    keep_last_n : int
        Número de sesiones recientes a mantener
    dry_run : bool
        Si True, solo muestra qué se eliminaría
    """
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experimento '{experiment_name}' no encontrado")
            return
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=""
        )
        
        # Agrupar por session_id
        sessions = {}
        for run in runs:
            session_id = run.data.tags.get('session_id', 'unknown')
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(run)
        
        # Ordenar sesiones por timestamp (más reciente primero)
        sorted_sessions = sorted(
            sessions.items(),
            key=lambda x: x[0],
            reverse=True
        )
        
        print(f"\n📁 Experimento: {experiment_name}")
        print(f"   Total sesiones: {len(sessions)}")
        print(f"   Mantener últimas: {keep_last_n}")
        
        sessions_to_delete = sorted_sessions[keep_last_n:]
        
        if sessions_to_delete:
            print(f"\n⚠️  Sesiones a eliminar ({len(sessions_to_delete)}):")
            
            total_runs_to_delete = 0
            for session_id, session_runs in sessions_to_delete:
                print(f"   - {session_id}: {len(session_runs)} runs")
                total_runs_to_delete += len(session_runs)
                
                if not dry_run:
                    for run in session_runs:
                        try:
                            client.delete_run(run.info.run_id)
                        except Exception as e:
                            print(f"     ⚠️  Error eliminando run: {e}")
            
            if dry_run:
                print(f"\n⚠️  DRY RUN - No se eliminó nada")
                print(f"   Se eliminarían {total_runs_to_delete} runs de {len(sessions_to_delete)} sesiones")
            else:
                print(f"\n✅ Eliminados {total_runs_to_delete} runs de {len(sessions_to_delete)} sesiones")
        else:
            print(f"   ✅ No hay sesiones para eliminar")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Menú interactivo para limpieza de MLflow"""
    
    # Configurar MLflow
    mlruns_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlruns'))
    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    
    print(f"\n{'='*100}")
    print(f"🧹 LIMPIEZA DE MLFLOW")
    print(f"{'='*100}")
    print(f"Tracking URI: {mlruns_path}")
    
    while True:
        print(f"\n{'─'*100}")
        print("Opciones:")
        print("  1. Listar todos los experimentos")
        print("  2. Eliminar runs fallidos/sin métricas (DRY RUN)")
        print("  3. Eliminar runs fallidos/sin métricas (EJECUTAR)")
        print("  4. Eliminar experimento completo")
        print("  5. Limpiar sesiones antiguas (mantener últimas N)")
        print("  6. Salir")
        print(f"{'─'*100}")
        
        choice = input("\nSelecciona opción (1-6): ").strip()
        
        if choice == '1':
            list_all_experiments()
        
        elif choice == '2':
            exp_name = input("\nNombre del experimento (Enter para todos): ").strip()
            exp_name = exp_name if exp_name else None
            delete_failed_runs(exp_name, dry_run=True)
        
        elif choice == '3':
            exp_name = input("\nNombre del experimento (Enter para todos): ").strip()
            exp_name = exp_name if exp_name else None
            confirm = input(f"⚠️  ¿Confirmar eliminación? (yes/no): ")
            if confirm.lower() == 'yes':
                delete_failed_runs(exp_name, dry_run=False)
            else:
                print("❌ Cancelado")
        
        elif choice == '4':
            exp_name = input("\nNombre del experimento a eliminar: ").strip()
            if exp_name:
                delete_experiment(exp_name, dry_run=False)
            else:
                print("❌ Nombre requerido")
        
        elif choice == '5':
            exp_name = input("\nNombre del experimento: ").strip()
            if exp_name:
                try:
                    keep_n = int(input("¿Cuántas sesiones mantener? (default: 5): ").strip() or "5")
                    dry_run = input("¿Dry run? (yes/no, default: yes): ").strip().lower() != 'no'
                    cleanup_old_sessions(exp_name, keep_last_n=keep_n, dry_run=dry_run)
                except ValueError:
                    print("❌ Número inválido")
            else:
                print("❌ Nombre requerido")
        
        elif choice == '6':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción inválida")


if __name__ == "__main__":
    main()
