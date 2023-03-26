import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("主成分分析")
    st.write("""PCA is a technique used to reduce the dimensionality of a data set. It projects the data into a lower-dimensional space while retaining most of the variation in the data.""")
    
    file = st.file_uploader("エクセルファイルをアップロードしてください。", type=["xlsx"])
    
    if file is not None:
        st.write("### データの読み込み")
        df = pd.read_excel(upload_files_xlsx, sheet_name=0)
        st.write(df.head())
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        pca = PCA()
        pca.fit(scaled_data)
        
        st.write("### 説明可能な分散比")
        st.line_chart(pca.explained_variance_ratio_)
        
        num_components = st.slider("成分数を選択", 1, len(df.columns))
        
        pca = PCA(n_components=num_components)
        principal_components = pca.fit_transform(scaled_data)
        
        columns = ["PC" + str(i) for i in range(1, num_components+1)]
        pc_df = pd.DataFrame(data=principal_components, columns=columns)
        
        st.write("### Principal Components")
        st.write(pc_df.head())
        st.write("説明可能な分散比:", sum(pca.explained_variance_ratio_))
        
        st.write("### Scatter Plot of First Two Principal Components")
        st.pyplot(plt.scatter(pc_df["PC1"], pc_df["PC2"]))
        
if __name__ == "__main__":
    main()
