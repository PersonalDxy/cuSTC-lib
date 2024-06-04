from flask import Flask, request, jsonify, send_file, send_from_directory
import numpy as np
import io
import zipfile
import tensorly as tl
from tensorly.decomposition import parafac
import random
import subprocess
import sys
import os
import json

# from decomp import main

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('./', 'index.html')

@app.route('/decompose', methods=['POST'])
def decompose():
    tensor_data = request.files['tensor']
    algorithm = request.form['algorithm']
    tensor_index = int(request.form['tensorIndex'])

    file_stream = io.StringIO(tensor_data.read().decode('utf-8'))

    original = []
    mini = sys.maxsize
    minj = sys.maxsize
    mink = sys.maxsize
    maxi = 0
    maxj = 0
    maxk = 0

    print(algorithm)

    for each in file_stream:
        otensorb = []
        # print(each)
        anewline = each.split(' ')
        otensorb.append(int(anewline[0]))
        otensorb.append(int(anewline[1]))
        otensorb.append(int(anewline[2]))
        otensorb.append(float(anewline[3]))
        if(otensorb[0]>maxi):
            maxi = otensorb[0]
        if(otensorb[1]>maxj):
            maxj = otensorb[1]
        if(otensorb[2]>maxk):
            maxk = otensorb[2]
        if(otensorb[0]<mini):
            mini = otensorb[0]
        if(otensorb[1]<minj):
            minj = otensorb[1]
        if(otensorb[2]<mink):
            mink = otensorb[2]
        original.append(otensorb)

    # original.sort(key=itemgetter(0,1))
    tensor = np.zeros((maxi,maxj,maxk), dtype=float)
    for each in original:
        tensor[each[0]-1][each[1]-1][each[2]-1] = each[3]

    tensor1 = tensor.reshape((maxi,maxj,maxk))
    rank = tensor_index

    script_dir = os.path.join(os.getcwd(), '/home/jxycdxy/MBLC/cuSTC/src/mblc/sgpu/')  # 指定脚本所在目录
    script_path = os.path.join(script_dir, 'run.sh')  # 脚本的完整路径

    if algorithm == 'als':
        # result = subprocess.run("sh /home/jxycdxy/MBLC/cuSTC/src/mblc/sgpu/run0.sh", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = subprocess.run("sh run0.sh", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # result = subprocess.run(['bash', 'run0.sh'], cwd=script_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        weight, factors = parafac(tensor1, rank)
        print(output)
        # print(factors_list)

        # factors_list = [tl.to_numpy(factor).tolist() for factor in factors]

        # print(rank)

        # print("------------------")

        # print(factors)

        # tensor = np.load(tensor_data, encoding='bytes', allow_pickle=True)
    elif algorithm == 'sgd':
         # result = subprocess.run("sh /home/jxycdxy/MBLC/cuSTC/src/mblc/sgpu/run0.sh", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = subprocess.run("sh /home/jxycdxy/MBLC/cuSTC/src/mblc/sgpu/run1.sh", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        weight, factors = parafac(tensor1, rank)
        print(output)
        # print(factors_list)

        # factors_list = [tl.to_numpy(factor).tolist() for factor in factors]

        # print(rank)

        # print("------------------")

        # print(factors)

        # tensor = np.load(tensor_data, encoding='bytes', allow_pickle=True)
    elif algorithm == 'gd':
         # result = subprocess.run("sh /home/jxycdxy/MBLC/cuSTC/src/mblc/sgpu/run0.sh", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = subprocess.run("sh /home/jxycdxy/MBLC/cuSTC/src/mblc/sgpu/run2.sh", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        weight, factors = parafac(tensor1, rank)
        print(output)
        # print(factors_list)

        # factors_list = [tl.to_numpy(factor).tolist() for factor in factors]

        # print(rank)

        # print("------------------")

        # print(factors)

        # tensor = np.load(tensor_data, encoding='bytes', allow_pickle=True)
    

    factors_list = [factor.tolist() for factor in factors]

    factors_file_path = os.path.join('./', 'factors_list.json')
    with open(factors_file_path, 'w') as f:
        json.dump(factors_list, f)

    # print(factors_list)

    # factors_list = [tl.to_numpy(factor).tolist() for factor in factors]

    # print(rank)

    # print("------------------")

    # print(factors)

    # tensor = np.load(tensor_data, encoding='bytes', allow_pickle=True)
    
    # Perform tensor decomposition
    # if algorithm == 'ALS':
    #     factors = tl.decomposition.parafac(tensor, rank=tensor_index)
    # elif algorithm == 'GD':
    #     factors = tl.decomposition.parafac(tensor, rank=tensor_index)
    # elif algorithm == 'SGD':
    #     factors = tl.decomposition.parafac(tensor, rank=tensor_index)
    # else:
    #     return jsonify({"error": "Invalid algorithm"})

    # Convert factors to list of numpy arrays
    # factors_list = [factor.tolist() for factor in factors]

    # Simulate speedup results
    speedupFCOO = {
        "labels": ['Dimension-1', 'Dimension-2', 'Dimension-3'],
        "data": [random.uniform(1, 1.2), random.uniform(2, 4), random.uniform(2, 4)]
    }
    speedupMMCSF = {
        "labels": ['Dimension-1', 'Dimension-2', 'Dimension-3'],
        "data": [random.uniform(1, 1.1), random.uniform(0.9, 4), random.uniform(0.9, 4)]
    }

    return jsonify({"result": factors_list, "speedupFCOO": speedupFCOO, "speedupMMCSF": speedupMMCSF})

@app.route('/download-factor-matrix')
def download_factor_matrix():
    return send_file(os.path.join('./', 'factors_list.json'), as_attachment=True)

@app.route('/complete', methods=['POST'])
def complete():
    tensor_data = request.files['tensor']
    algorithm = request.form['algorithm']
    rank = int(request.form['rank'])

    file_stream = io.StringIO(tensor_data.read().decode('utf-8'))

    original = []
    mini = sys.maxsize
    minj = sys.maxsize
    mink = sys.maxsize
    maxi = 0
    maxj = 0
    maxk = 0

    for each in file_stream:
        otensorb = []
        # print(each)
        anewline = each.split(' ')
        otensorb.append(int(anewline[0]))
        otensorb.append(int(anewline[1]))
        otensorb.append(int(anewline[2]))
        otensorb.append(float(anewline[3]))
        if(otensorb[0]>maxi):
            maxi = otensorb[0]
        if(otensorb[1]>maxj):
            maxj = otensorb[1]
        if(otensorb[2]>maxk):
            maxk = otensorb[2]
        if(otensorb[0]<mini):
            mini = otensorb[0]
        if(otensorb[1]<minj):
            minj = otensorb[1]
        if(otensorb[2]<mink):
            mink = otensorb[2]
        original.append(otensorb)

    # original.sort(key=itemgetter(0,1))
    tensor = np.zeros((maxi,maxj,maxk), dtype=float)
    for each in original:
        tensor[each[0]-1][each[1]-1][each[2]-1] = each[3]

    tensor1 = tensor.reshape((maxi,maxj,maxk))
    rank = rank

    script_dir = os.path.join(os.getcwd(), '/home/jxycdxy/MBLC/cuSTC/src/mblc/sgpu/')  # 指定脚本所在目录
    script_path = os.path.join(script_dir, 'run.sh')  # 脚本的完整路径

    if algorithm == 'als':
        # result = subprocess.run("sh /home/jxycdxy/MBLC/cuSTC/src/mblc/sgpu/run0.sh", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = subprocess.run("sh run0.sh", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # result = subprocess.run(['bash', 'run0.sh'], cwd=script_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        weight, factors = parafac(tensor1, rank)
        print(output)
        # print(factors_list)

        # factors_list = [tl.to_numpy(factor).tolist() for factor in factors]

        # print(rank)

        # print("------------------")

        # print(factors)

        # tensor = np.load(tensor_data, encoding='bytes', allow_pickle=True)
    elif algorithm == 'sgd':
         # result = subprocess.run("sh /home/jxycdxy/MBLC/cuSTC/src/mblc/sgpu/run0.sh", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = subprocess.run("sh /home/jxycdxy/MBLC/cuSTC/src/mblc/sgpu/run1.sh", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        weight, factors = parafac(tensor1, rank)
        print(output)
        # print(factors_list)

        # factors_list = [tl.to_numpy(factor).tolist() for factor in factors]

        # print(rank)

        # print("------------------")

        # print(factors)

        # tensor = np.load(tensor_data, encoding='bytes', allow_pickle=True)
    elif algorithm == 'gd':
         # result = subprocess.run("sh /home/jxycdxy/MBLC/cuSTC/src/mblc/sgpu/run0.sh", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = subprocess.run("sh /home/jxycdxy/MBLC/cuSTC/src/mblc/sgpu/run2.sh", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        weight, factors = parafac(tensor1, rank)
        print(output)
        # print(factors_list)

        # factors_list = [tl.to_numpy(factor).tolist() for factor in factors]

        # print(rank)

        # print("------------------")

        # print(factors)

        # tensor = np.load(tensor_data, encoding='bytes', allow_pickle=True)

    # factors_list = [factor.tolist() for factor in factors]
    factors_list = factors

    # I, R = factors_list[0].shape
    # J, _ = factors_list[1].shape
    # K, _ = factors_list[2].shape
    I = maxi
    J = maxj
    K = maxk
    R = rank

    X_hat = np.zeros((I, J, K))

    # 使用因子矩阵重构张量
    for r in range(R):
        X_hat += np.outer(np.outer(factors_list[0][:, r], factors_list[1][:, r]), factors_list[2][:, r]).reshape((I, J, K))

    np.savetxt('completed_tensor.txt', X_hat.flatten())

    # Simulate speedup results
    speedupFCOO = {
        "labels": ['Dimension-1', 'Dimension-2', 'Dimension-3'],
        "data": [1.5, 1.7, 2.0]
    }
    speedupMMCSF = {
        "labels": ['Dimension-1', 'Dimension-2', 'Dimension-3'],
        "data": [1.2, 1.4, 1.6]
    }

    # Perform completion logic here
    result = {"example_result": "result_data"}  # Placeholder result

    return jsonify({"result": result, "speedupFCOO": speedupFCOO, "speedupMMCSF": speedupMMCSF})

@app.route('/download-completed-tensor')
def download_completed_tensor():
    return send_file(os.path.join('./', 'completed_tensor.txt'), as_attachment=True)

@app.route('/nn_decompose', methods=['POST'])
def nn_decompose():
    tensor = request.files['tensor']
    algorithm = request.form['algorithm']
    rank = int(request.form['rank'])

    # Simulate decomposition and accuracy/compression results
    accuracy = {
        "labels": [4, 16, 64, 256],
        "data": [0.75, 0.82, 0.88, 0.91]
    }
    compression = {
        "labels": [4, 16, 64, 256],
        "data": [40, 30, 20, 10]
    }

    # command = [
    #     'python3', 'scripts/decomp.py',
    #     '-p', tensor.filename,
    #     '-d', algorithm,
    #     '-r', str(rank)
    # ]
    # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # output, error = process.communicate()

    # # 解析分解结果
    # accuracy, compression = parse_output(output)

    # result = main(tensor, algorithm, rank)

    return jsonify({"accuracy": accuracy, "compression": compression})

@app.route('/export_model', methods=['POST'])
def export_model():
    # Create a sample model file
    model_data = io.BytesIO()
    with zipfile.ZipFile(model_data, 'w') as zf:
        zf.writestr('model.txt', 'This is a sample model file.')

    model_data.seek(0)
    return send_file(model_data, attachment_filename='model.zip', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10855, debug=True)
