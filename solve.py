import os
import stat
import json
import time
import tqdm
import numpy as np

import torch
import torch.nn as nn

from datetime import datetime
from petsc4py import PETSc
from scipy.sparse import diags


########################################################
# 1. SOLVER
########################################################

def build_folder(exp_id, dataset, solver_array, precond_array, tol_array, max_iter, size_mat, num_data):

    # current_date = datetime.now().strftime('%Y%m%d')  # YYYYMMDD format
    current_date = exp_id

    if not os.path.exists('./results'):
        os.makedirs('./results')


    rel_path = './results/data_{}_{}_{}'.format(dataset, num_data, current_date)

    if not os.path.exists(rel_path):
        os.makedirs(rel_path)


    if not os.path.exists(rel_path + '/data_{}_{}_{}'.format(dataset, size_mat, num_data)):
        os.makedirs(rel_path + '/data_{}_{}_{}'.format(dataset, size_mat, num_data))

    if not os.path.exists(rel_path + '/data_{}_{}_{}_PETSc'.format(dataset, size_mat, num_data)):
        os.makedirs(rel_path + '/data_{}_{}_{}_PETSc'.format(dataset, size_mat, num_data))

    for solver in solver_array:
        for precond in precond_array:
            for tol in tol_array:
                if not os.path.exists(
                        rel_path + '/data_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                       size_mat,
                                                                       num_data)):
                    os.makedirs(
                        rel_path + '/data_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                       size_mat,
                                                                       num_data))

    if not os.path.exists(rel_path + '/output'):
        os.makedirs(rel_path + '/output')

    for solver in solver_array:
        for precond in precond_array:
            for tol in tol_array:
                if not os.path.exists(
                        rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                                size_mat,
                                                                                num_data)):
                    os.makedirs(
                        rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                                size_mat,
                                                                                num_data))
                if not os.path.exists(
                        rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}/total'.format(dataset, solver, precond, tol,
                                                                                      max_iter,
                                                                                      size_mat,
                                                                                      num_data)):
                    os.makedirs(
                        rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}/total'.format(dataset, solver, precond, tol,
                                                                                      max_iter,
                                                                                      size_mat,
                                                                                      num_data))


    if not os.path.exists(rel_path + '/seq'):
        os.makedirs(rel_path + '/seq')

    if not os.path.exists(rel_path + '/results'):
        os.makedirs(rel_path + '/results')

    return rel_path

def build_helmholtz(coef, P):
    K = coef.shape[0]
    P = (P.numpy())**2
    s = K - 2

    P = np.reshape(P,s*s,order='C')

    diag_list = []
    off_diag_list = []
    for j in range(1, K-1):
        diag_values = np.array([
            np.concatenate((-0.5 * (coef[1:K-2, j] + coef[2:K-1, j]),[0])),
            0.5 * (coef[0:K-2, j] + coef[1:K-1, j]) + 0.5 * (coef[2:K, j] + coef[1:K-1, j]) + \
            0.5 * (coef[1:K-1, j-1] + coef[1:K-1, j]) + 0.5 * (coef[1:K-1, j+1] + coef[1:K-1, j]),
            np.concatenate((-0.5 * (coef[1:K-2, j] + coef[2:K-1, j]),[0]))
        ])
        diag_list.append(diag_values)

        if j != K-2:

            off_diag = -0.5 * (coef[1:K-1, j] + coef[1:K-1, j+1])
            off_diag_list.append(off_diag)

    diag_output = np.concatenate(diag_list,axis=1)
    off_diag_output = np.concatenate(off_diag_list,axis=0)
    A = (diags(diag_output,[-1,0,1],(s**2,s**2)) + diags((off_diag_output,off_diag_output),[-(K-2),(K-2)],(s**2,s**2))) * (K-1)**2 + diags(P,0,(s**2,s**2))

    return A

def rebulid_A(parameters, size_mat, test_num:int, data_name:str, is_b_static=True,random_scale=None):
    shape = parameters.shape

    if test_num > shape[0]:
        len = shape[0]
        print(f"Test number {test_num} is large than the number examples {shape[0]}")
    else:
        len = test_num
        print(f"Using {len} Examples")

    A = []
    cof = np.ones((shape[1]+2,shape[2]+2))

    if is_b_static:
        b = np.ones((len,size_mat,1))
    else:
        b = random_scale*np.random.randn((len,size_mat,1))
    
    pbar = tqdm.tqdm(desc=f'Rebulid Matrix A({size_mat}*{size_mat})',total=len,leave=True,position=0)
    for i in range(len):
        pbar.update()
        p = parameters[i].squeeze()
        a = build_helmholtz(coef=cof,P=p)
        A.append(a)

    pbar.close()
    return A,b

def petsc_generator(dataset, size_mat, num_data, rel_path, A, b, U):
    s = int(np.sqrt(size_mat))

    for i in range(num_data):
        A0 = A[i]
        b0 = b[i]
        U0 = U[i].numpy().squeeze()

        A_csr = A0.tocsr()

        A_petsc = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
        b_petsc = PETSc.Vec().createWithArray(b0, comm=PETSc.COMM_WORLD)


        U_p = PETSc.Mat().createDense(size=U0.shape,array=U0)

        viewer_A = PETSc.Viewer().createBinary(
            rel_path + '/data_{}_{}_{}_PETSc/'.format(dataset, size_mat, num_data) + 'A_%d.dat' % i, 'w')
        viewer_b = PETSc.Viewer().createBinary(
            rel_path + '/data_{}_{}_{}_PETSc/'.format(dataset, size_mat, num_data) + 'rhs_%d.dat' % i, 'w')
        viewer_U = PETSc.Viewer().createBinary(
            rel_path + '/data_{}_{}_{}_PETSc/'.format(dataset, size_mat, num_data) + 'U_%d.dat' % i, 'w')

        A_petsc.view(viewer_A)
        b_petsc.view(viewer_b)
        U_p.view(viewer_U)


        viewer_A.destroy()
        viewer_b.destroy()
        viewer_U.destroy()


    return 0

def record_experiment_parameters(exp_id, theme, rel_path, dataset, solver_array, precond_array,
                                 tol_array, max_iter, size_mat, num_data):

    dir_results = rel_path
    now = datetime.now()
    with open(dir_results + '/parameters.txt', 'a') as file:
        file.write('theme: ' + str(theme) + '\n')
        file.write('exp_id: ' + str(exp_id) + '\n')
        file.write('exp_time: ' + str(now) + '\n')
        file.write('dataset: ' + str(dataset) + '\n')
        file.write('solver_array: ' + str(solver_array) + '\n')
        file.write('precond_array: ' + str(precond_array) + '\n')
        file.write('tol_array: ' + str(tol_array) + '\n')
        file.write('max_iter: ' + str(max_iter) + '\n')
        file.write('size_mat: ' + str(size_mat) + '\n')
        file.write('num_data: ' + str(num_data) + '\n')
        file.close()
    return 0


def record_experiment_start(dir_results, cmd, dataset, solver, precond, tol, max_iter, size_mat,
                                                               num_data):
    now = datetime.now()
    with open(dir_results + '/exp_record.txt', 'a') as file:
        output = "exp_start: {}, {}, {}, {}, {}, {}, {}\n".format(dataset, solver, precond, tol, max_iter, size_mat,
                                                               num_data)
        file.write(output)
        file.write('exp_start_cmd: ' + str(cmd) + '\n')
        file.write('exp_start_time: ' + str(now) + '\n')
        file.close()
    return 0

def record_experiment_end(dir_results, dataset, solver, precond, tol, max_iter, size_mat,
                                                               num_data):
    now = datetime.now()
    with open(dir_results + '/exp_record.txt', 'a') as file:
        file.write('exp_end_time: ' + str(now) + '\n')
        output = "exp_end: {}, {}, {}, {}, {}, {}, {}\n".format(dataset, solver, precond, tol, max_iter, size_mat,
                                                               num_data)
        file.write(output)
        file.close()
    return 0

def petsc_solver(dataset, size_mat, num_data, rel_path, solver_array, precond_array, tol_array, max_iter, mat_dim, k_dim):
    dir = rel_path + '/data_{}_{}_{}_PETSc'.format(dataset, size_mat, num_data)
    dir = '/PETSc/no4mat/results/data_helmholtz_64_test/data_helmholtz_62500_64_PETSc'
    dir_results = rel_path + '/results'

    for solver in solver_array:
        for precond in precond_array:
            precond_cmd = precond
            # cmd_run = "mpirun --allow-run-as-root -n 72 ./e"
            cmd_run = "sudo mpirun --allow-run-as-root -n 16 ./e" # If you are not root, change the cmd.
            # cmd_run = "./e"
            if precond == 'cholesky':
                cmd_run = "./e"
            elif precond == 'ilu0':
                cmd_run = "./e"
                precond_cmd = "ilu -pc_factor_levels 0"
            elif precond == 'ilu1':
                cmd_run = "./e"
                precond_cmd = "ilu -pc_factor_levels 1"
            elif precond == 'ilu2':
                cmd_run = "./e"
                precond_cmd = "ilu -pc_factor_levels 2"
            elif precond == 'eisenstat':
                precond_cmd = "sor -pc_sor_variant eisenstat"
            elif precond == 'icc0':
                cmd_run = "./e"
                precond_cmd = "icc -pc_factor_levels 0"
            elif precond == 'icc1':
                cmd_run = "./e"
                precond_cmd = "icc -pc_factor_levels 1"
            elif precond == 'icc2':
                cmd_run = "./e"
                precond_cmd = "icc -pc_factor_levels 2"

            for tol in tol_array:

                if solver == 'gmres':
                    dir_x = rel_path + '/data_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                           size_mat, num_data)
                    dir_output = rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol,
                                                                                         max_iter,
                                                                                         size_mat, num_data)
                    cmd = cmd_run + ' -ksp_converged_reason -pc_type {} -ksp_rtol {} -ksp_gmres_restart 40 -ksp_type hpddm ' \
                          '-nmat {} -load_dir {} -load_dir_x {} -load_dir_output ' \
                          '{} -ksp_max_it {} -mat_dim {} -k_dim {}' \
                        .format(precond_cmd, tol, num_data, dir, dir_x, dir_output, max_iter, mat_dim, k_dim)

                if solver == 'gcrodr':
                    dir_x = rel_path + '/data_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                           size_mat,num_data)
                    dir_output = rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                           size_mat,num_data)

                    cmd = cmd_run + ' -ksp_converged_reason -pc_type {} -ksp_rtol {} -ksp_gmres_restart 40 -ksp_type hpddm ' \
                          '-ksp_hpddm_type {} -ksp_hpddm_recycle 20 -nmat {} -load_dir {} -load_dir_x {} -load_dir_output ' \
                          '{} -ksp_max_it {} -mat_dim {} -k_dim {}'\
                        .format(precond_cmd, tol, solver, num_data, dir, dir_x, dir_output, max_iter, mat_dim, k_dim)

                print(cmd)
                record_experiment_start(dir_results, cmd, dataset, solver, precond, tol, max_iter, size_mat,
                                       num_data)

                start_time = time.perf_counter()
                os.system(cmd)
                end_time = time.perf_counter()
                
                record_experiment_end(dir_results, dataset, solver, precond, tol, max_iter, size_mat, num_data)

                total_time = end_time - start_time
                average_time = total_time / num_data
                rel_path_total = rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}/total'.format(dataset, solver, precond, tol, max_iter,
                                                                           size_mat,num_data)

                with open(rel_path_total + '/total_time.txt', 'w') as file:
                    file.write(str(total_time))
                    file.close()
                with open(rel_path_total + '/average_time.txt', 'w') as file:
                    file.write(str(average_time))
                    file.close()

                total_iter = []
                with open(rel_path_total + '/output_total_iter.txt', 'r') as file:
                    for line in file:
                        iter = int(line.strip())
                        total_iter.append(iter)
                    file.close()
                max_iter_count = total_iter.count(max_iter)
                with open(rel_path_total + '/max_iter_count.txt', 'w') as file:
                    file.write(str(max_iter_count))
                    file.close()
                average_iter = np.mean(total_iter)
                with open(rel_path_total + '/average_iter.txt', 'w') as file:
                    file.write(str(average_iter))
                    file.close()

                print(dataset, solver, precond, tol, max_iter, size_mat, num_data, 'done')
    return 0

########################################################
# 2. SOLVING
########################################################
if __name__ == "__main__":

    size_mat = 62500
    num_data = 64
    k_dim = 10
    exp_id = 'test'
    max_iter = size_mat
    theme = 'test'
    # precond_array = ["none", "jacobi", "bjacobi", "sor", "asm", "icc0", "ilu0"]
    precond_array = ["none"]
    # tol_array = [1e-2, 1e-3, 1e-4, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12]
    tol_array = [1e-2]
    solver_array = ["gcrodr", "gmres"]

    data_path = './results/'
    dataset = 'helmholtz'
    rel_path = './results/data_{}_{}_{}'.format(dataset, num_data, exp_id)
    saved_dict = torch.load(data_path+'preds.pt')

    params = saved_dict['params']
    subspace = saved_dict['preds']
    print('*'*20+' RECONSTRUCTION '+'*'*20)

    build_folder(exp_id, dataset, solver_array, precond_array, tol_array, max_iter, size_mat, num_data)

    record_experiment_parameters(exp_id, theme, rel_path, dataset, solver_array, precond_array,
                                 tol_array, max_iter,
                                 size_mat, num_data)
    
    A, b = rebulid_A(parameters=params, size_mat=size_mat, test_num=num_data, data_name=dataset)

    petsc_generator(dataset, size_mat, num_data, rel_path, A, b, subspace)

    print('*'*20+' SOLVING '+'*'*20)
    petsc_solver(dataset, size_mat, num_data, rel_path, solver_array, precond_array, tol_array, max_iter,size_mat,k_dim)
