이번 

복습 필요한 사항
☆ Target encoding , Data leakage 에 대한 개념이 아직 모호함. 
☆ OOF 에 대해서 개념 모호함. 

1. display technique

2. os module

3. difference , intersection

4. 시계열 데이터 plot

5. 승차 & 하차 에 관한 columns 선별 insight

6. msno.matrix ( missing value checking technique )

7. folium 에 대해 간략히 다뤄서 공간지도 데이터 handling

8. null value 는 base line 에서는 그대로 사용하는 것이 좋다
--> 비선형모델 ( tree-based model ) 에서는 nullvalue 도 학습하기에 이후에 해석되는 걸 토대로 변형하는것이 훨씬 효율적임.

9. 정규화 log1p , sqrt , box-cox 

10. label encoder [transform , inverse transform]

11. pdp plot ; Showing margin effect 

12. baseline ; [선형] Ridge , Lasso [비선형] Random Forest , Decision Tree , LightGBM 이상 5가지를 토대로 진행하여봄
이 때 linear은 coefficient , non-linear 은 feature importance 를 토대로 1차적으로 해석해봄.
2차적으로는 linear 에서 잘 나오지 않았으나 non-linear에서 잘 나온 feature 라면 NN ( deep learning ) 으로 해봄으로써 성능 향상에 대한 가설을 수립할 수 있음.

13. 날짜 핸들링 시 'datetime / str' 확인한다.
만약 str 일시 pd.to_datetime 으로 변환해줘서 df.month ~ 등 여러 함수로 좀 더 원활하게 날짜 데이터를 핸들링해보자 .
+ dtype , astype 와 친해지기.

14. Feature selection / elimination
1) Bortua
2) Target Permutation
3) Dropping High Correlation
4) RFE(Recursive Feature Elimination)

15. ENSEMBLE / STACKING
1) Ensemble 각 나온 결과치를 토대로 각 모델에 가중치를 주어 submission 
2) Stacking , 각 모델로부터 나온 결과를 다시 학습하여 다양성(variance) 을 올림으로써 성능향상기대하고자 함.





