Model : SVR

Data : Self Datasets

Comment : SVR and NuSVR implematation using sklearn and compare to other models

- NuSVR : soft margine을 사용하는 일반적인 SVM 모델들은 support vector를 선택함에 있어,
              C 라는 하이퍼파라미터를 사용하기 때문에 decision boundary를 결정할 때 발생하는 
              오차들을 컨트롤 할 수 있다. 
              반면, NuSVM은 nu 라는 하이퍼파라미터를 이용하여, support vector의 수를 컨트롤 할 수 있다.
              SVR의 관점으로 보면, tube 바깥의 data point 수를 컨트롤 함으로써 tube의 크기를 컨트롤 하는 
              것으로 볼 수 있다.   