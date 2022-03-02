import pandas as pd
import os

"""
    csv_data/1/*.csv
    위의 경로에 모든 모델의 결과 값을 저장해 놓는다.
    위의 경로는 사용자가 편한 경로로 바꾸면 된다. 경로를 수정할 경우 아래의 file_path 경로를 알맞게 수정해야 한다.

    ### Voting Classification ###
    num : 몇개의 모델에서 이상치라고 판단했을 때 이상치라고 정할지를 결정하는 변수이다. Ex) 3이라는 숫자를 변수에 넣으면 3개의 모델에서 동시에 이상치라고
    판단한 부분을 실제 이상치라고 판단한다. 만약에 2개의 모델에서 동시에 이상치라고 판단했어도 이러한 부분은 이상치라고 판단하지 않고 버린다.

    version : csv 결과 파일을 모아 놓는 경로이다. 여기에서 버전 별로 구분하기 위해서 독립적인 파일에 결과 파일을 저장 했다.
"""
num = 3
version = '1'
file_path = './/csv_data//' + version

file_names = os.listdir(file_path)[:-1]
print(file_names)

base = pd.read_csv(os.path.join(file_path, file_names[0]))

print(base['attack'].value_counts())
for file_name in file_names[1:]:
    print(file_name)
    temp = pd.read_csv(os.path.join(file_path, file_name))

    base['attack'] += temp['attack']

print(base['attack'].value_counts())

"""
    ### Voting Classification ###
    여기 부분에서 실제로 Voting Classification이 일어난다.
    위에서는 Voting을 위해서 결과 값들의 합산을 구한다.   
"""
if num > 1:
    base.loc[base['attack'] < num, 'attack'] = 0

base.loc[base['attack'] >= num, 'attack'] = 1
print(base['attack'].value_counts())

"""
    기본적으로 윈도우 사이즈가 다른 6개의 모델의 합산을 구한다.
    그리고 Resnet모델은 이렇게 구해진 결과 값에 그대로 더해주기만 하면 된다. 
"""
base.to_csv('version1_num_3.csv', index=False)

"""
    위의 과정을 통해서 Voting이 끝났다. 이제는 Resnet+GRU 모델에서 학습한 결과 값을 더해준다.
    Resnet+GRU 모델을 Voting에 넣지 않고 합산하는 이유는 이 모델 같은 경우는 넓은 범위의 이상치를 감지하는 것에 좋은 성능을 보이기 때문에
    필터링 작업을 거쳐서 짧은 범위의 이상치를 판별하는 것을 모두 제거했기 때문이다.

    a : 위에서 voting한 결과 값을 변수에 저장한다.
    b : Resnet+GRU 모델의 결과 값을 변수에 저장한다.
"""
a = pd.read_csv('version1_num_3.csv')
b = pd.read_csv('csv_data//3//resnet.csv')

a.iloc[b.loc[b.attack == 1].index, 1] = 1

"""
    최종적인 결과 값이 나온다. 
"""
a.to_csv('version1_num_3(+resnet).csv', index=False)
