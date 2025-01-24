#include <petsc.h>

static char help[] = "Solves a series of linear systems using KSPHPDDM.\n\n";

// int gcrodr(int argc, char **args)
int main(int argc, char **args)
{
  Vec x, b; /* computed solution and RHS */
  Mat A;    /* linear system matrix */
  KSP ksp;  /* linear solver context */

  PetscInt i, j, nmat = 2;
  PetscViewer viewer;
  char dir[PETSC_MAX_PATH_LEN], dir_output[1000], dir_x[1000], name[256];
  PetscBool flg, reset = PETSC_FALSE;

  PetscInt iter;
  PetscReal rnorm;
  char filename[PETSC_MAX_PATH_LEN]; // 假设文件名最长50个字符
  char filename_total_iter[PETSC_MAX_PATH_LEN];
  char filename_total_rnorm[PETSC_MAX_PATH_LEN];

  PetscViewer filex; // 输出结果

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscStrncpy(dir, ".", sizeof(dir)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-load_dir", dir, sizeof(dir), NULL));

  PetscCall(PetscStrncpy(dir_output, ".", sizeof(dir_output)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-load_dir_output", dir_output, sizeof(dir_output), NULL));

  PetscCall(PetscStrncpy(dir_x, ".", sizeof(dir_x)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-load_dir_x", dir_x, sizeof(dir_x), NULL));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nmat", &nmat, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-reset", &reset, NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));

  snprintf(filename_total_iter, PETSC_MAX_PATH_LEN, "%s/total/output_total_iter.txt", dir_output); // 构造文件名
  FILE *file_total_iter = fopen(filename_total_iter, "w");
  snprintf(filename_total_rnorm, PETSC_MAX_PATH_LEN, "%s/total/output_total_rnorm.txt", dir_output); // 构造文件名
  FILE *file_total_rnorm = fopen(filename_total_rnorm, "w");

  for (i = 0; i < nmat; i++)
  {
    j = i ;
    PetscCall(PetscSNPrintf(name, sizeof(name), "%s/A_%" PetscInt_FMT ".dat", dir, j));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
    PetscCall(MatLoad(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    if (i == 0)
      PetscCall(MatCreateVecs(A, &x, &b));
    PetscCall(PetscSNPrintf(name, sizeof(name), "%s/rhs_%" PetscInt_FMT ".dat", dir, j));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &viewer));
    PetscCall(VecLoad(b, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(PetscObjectTypeCompare((PetscObject)ksp, KSPHPDDM, &flg));

    // 输出到文件
    snprintf(filename, PETSC_MAX_PATH_LEN, "%s/output_%d.txt", dir_output, j); // 构造文件名
    FILE *file = fopen(filename, "w");
    if (file != NULL)
    {
      KSPGetIterationNumber(ksp, &iter);
      KSPGetResidualNorm(ksp, &rnorm);
      fprintf(file, "Iterations: %d\n", iter);
      fprintf(file, "Residual norm: %g\n", rnorm);
      fclose(file);
      fprintf(file_total_iter, "%d\n", iter);
      fprintf(file_total_rnorm, "%g\n", rnorm);
    }
    else
    {
      PetscPrintf(PETSC_COMM_WORLD, "Error opening output file\n");
    }

    // 保存求解结果向量x为二进制文件
    snprintf(filename, PETSC_MAX_PATH_LEN, "%s/x_%d.dat", dir_x, j); // 构造文件名
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &filex);
    VecView(x, filex);
    PetscViewerDestroy(&filex);

  }

  fclose(file_total_iter);
  fclose(file_total_rnorm);

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -nmat 1 -pc_type none -ksp_converged_reason -ksp_type {{gmres hpddm}shared output} -ksp_max_it 1000 -ksp_gmres_restart 1000 -ksp_rtol 1e-10 -ksp_hpddm_type {{gmres bgmres}shared output} -options_left no -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR

   test:
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      suffix: 1_icc
      nsize: 1
      args: -nmat 1 -pc_type icc -ksp_converged_reason -ksp_type {{gmres hpddm}shared output} -ksp_max_it 1000 -ksp_gmres_restart 1000 -ksp_rtol 1e-10 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR

   testset:
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -nmat 3 -pc_type none -ksp_converged_reason -ksp_type hpddm -ksp_max_it 1000 -ksp_gmres_restart 40 -ksp_rtol 1e-10 -ksp_hpddm_type {{gcrodr bgcrodr}shared output} -ksp_hpddm_recycle 20 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR
      test:
        nsize: 1
        suffix: 2_seq
        output_file: output/ex75_2.out
      test:
        nsize: 2
        suffix: 2_par
        output_file: output/ex75_2.out

   testset:
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      nsize: 1
      args: -nmat 3 -pc_type icc -ksp_converged_reason -ksp_type hpddm -ksp_max_it 1000 -ksp_gmres_restart 40 -ksp_rtol 1e-10 -ksp_hpddm_type {{gcrodr bgcrodr}shared output} -ksp_hpddm_recycle 20 -reset {{false true}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR
      test:
        suffix: 2_icc
        args:
      test:
        suffix: 2_icc_atol
        args: -ksp_atol 1e-12

   test:
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      nsize: 2
      suffix: symmetric
      args: -nmat 3 -pc_type jacobi -ksp_converged_reason -ksp_type hpddm -ksp_max_it 1000 -ksp_gmres_restart 40 -ksp_atol 1e-11 -ksp_hpddm_type bgcrodr -ksp_hpddm_recycle 20 -reset {{false true}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR -ksp_hpddm_recycle_symmetric true

TEST*/
