import numpy as np
import os


def read_folders(path):
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            if ".bin" in file:
                yield file


def read_file(path):
    for file in os.listdir(path):
        if ".bin" in file:
            yield file


# output_file = read_file(os.getcwd())
output_file = 'series_data.npy'

# read output_file
data_output = np.load(output_file)  # , allow_pickle=True)
print(data_output)

# # data
# output = []
# for kk, file_tmp in enumerate(files(os.getcwd())):
#     print(file_tmp)
#     data_tmp = np.loadtxt(file_tmp)
#     # sortindex = np.argsort(data_tmp[:, 1])
#     # print "energy", data_tmp[:,2], "min", sortindex[0]
#     # data_tmp = data_tmp[sortindex[0],:]
#     output.append(data_tmp)
#
# output = np.array(output)
# # print(output)
# sort_param_index = 1
# sortindex = np.argsort(output[:, sort_param_index])
# output = output[sortindex, :]
# print(output[:, sort_param_index])
#
# lam_max = output[-1, 1]
# omega = output[-1, 0]
#
# np.savetxt('PdHF_om_%1.2f_nsites_4_lam_max_%1.2f_txt' % (omega, lam_max), output, fmt='%.4e', delimiter='\t')
