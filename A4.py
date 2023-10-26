import os
import sys
import subprocess

def run_task(task_file, resource_dir):
    subprocess.run(['python', task_file, resource_dir])

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python A4.py <RESOURCE_DIR> <A/B>.<TASK#>")
        sys.exit(1)

    resource_dir = sys.argv[1]
    task_id = sys.argv[2]

    if not os.path.exists(resource_dir):
        print(f"Resource directory {resource_dir} does not exist.")
        sys.exit(1)

    # Mapping task identifiers to their respective file names

    task_files = {
        'A.1': 'rosenbrock.py',
        'A.2': 'gradient_descent.py',
        'A.3': 'line_search.py',
        'A.4': 'newtons_method.py',
        'A.5': 'BFGS.py',
        'B.1': 'IK_1.py',
        'B.2': 'IK_2.py',
        'B.3': 'IK_4.py',
        'B.4': 'IK_n.py',
        'B.5': 'IK_n_Newton.py',
    }

    task_file = task_files.get(task_id)
    if task_file:
        run_task(resource_dir + '/' + task_file, resource_dir)
    else:
        print(f"Unknown task {task_id}. Valid tasks are: " + ", ".join(task_files.keys()))
        sys.exit(1)
