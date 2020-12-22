from inflearn 

# 01 section


###keyword ; pandas , seaborn , groupby , pivot_table , melt , unstack , ... etc


# 02 section

what is include this ipynb ?

ㆍcolumn 2개 이상 가져올 때 list 형식으로 가져와야함.

ㆍDataframe 2차원 , series 1차원

ㆍbins는 그 축 내에 얼마나 촘촘히 그릴것인가에 대해 설정해주는 파라미터

ㆍ피어슨 상관계수 - x,y가 함께 변하는 정도 / x,y가 각각 변하는 정도

ㆍ heatmap 그릴 때 대각선 제외하고 그리는 technique 도 포함되어 있음 ( np.triu)

ㆍ value_counts 는 series datatype에만 사용할 수 있음.

ㆍ서브셋 나눌 때 (2개 이상의 조건을 주는 것) 일 때 , () 괄호로 우선순위를 부여해줌. (df.a['X'] == 'asdf' & df.a['X] == 'naul')

ㆍdf.rename(columns= {}:{})

ㆍsns.barplot -> sns.carplot 으로 그리는 이유 ; col별로 그릴 수 있음 , 이 때 파라미터는 col , col_wrap

ㆍ빈도수 확인 할 때 .head / 혹은 특정 값 이상 확인 할 때 > OOO

##more study 

###isin을 사용해서 서브셋 만들기 ##
###unstack , stack , melt  
###sns plotting 과 pandas plotting 에 대해 좀 더 고민해보기.
