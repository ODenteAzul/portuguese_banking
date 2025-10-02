import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuração de visualização
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class BankDataEDA:
    """
    Classe para Análise Exploratória de Dados do dataset bancário.

    Arquitetura OO para organizar o código e facilitar reutilização.
    """

    def __init__(self, filepath: str, delimiter: str = ';'):
        """
        Inicializa o objeto e carrega os dados.

        Args:
            filepath: Caminho do arquivo CSV
            delimiter: Delimitador do CSV (padrão: ';')
        """
        self.df = pd.read_csv(filepath, delimiter=delimiter)
        self.numeric_cols = self.df.select_dtypes(
            include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(
            include=['object']).columns.tolist()

    def info_basica(self) -> None:
        """Mostra informações básicas do dataset."""
        print("=" * 60)
        print("INFORMAÇÕES BÁSICAS DO DATASET")
        print("=" * 60)
        print(
            f"""\nShape: {self.df.shape[0]:,} 
            linhas × {self.df.shape[1]} colunas""")
        print(
            f"Memória: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(
            f"\nVariáveis Numéricas ({len(self.numeric_cols)}): {', '.join(self.numeric_cols)}")
        print(
            f"Variáveis Categóricas ({len(self.categorical_cols)}): {', '.join(self.categorical_cols)}")

        print(f"\n HEAD: {self.df.head()}\n")

    def analise_missings(self) -> pd.DataFrame:
        """
        Analisa valores faltantes e 'unknown'.

        Returns:
            DataFrame com estatísticas de valores faltantes
        """
        print("\n" + "=" * 60)
        print("ANÁLISE DE VALORES FALTANTES E 'UNKNOWN'")
        print("=" * 60 + "\n")

        missing_stats = []

        for col in self.df.columns:
            # Valores nulos tradicionais
            null_count = self.df[col].isnull().sum()

            # Valores 'unknown' (comum em datasets categóricos)
            unknown_count = (self.df[col] == 'unknown').sum(
            ) if self.df[col].dtype == 'object' else 0

            total_missing = null_count + unknown_count
            percent = (total_missing / len(self.df)) * 100

            if total_missing > 0:
                missing_stats.append({
                    'coluna': col,
                    'nulls': null_count,
                    'unknowns': unknown_count,
                    'total_missing': total_missing,
                    'percentual': percent
                })

        if missing_stats:
            df_missing = pd.DataFrame(missing_stats).sort_values(
                'total_missing', ascending=False)
            print(df_missing.to_string(index=False))

            # Alerta para colunas críticas
            criticos = df_missing[df_missing['percentual'] > 20]
            if not criticos.empty:
                print(
                    f"\nATENÇÃO: {len(criticos)} coluna(s) com >20% de valores faltantes!")
        else:
            print("Nenhum valor faltante detectado!")

        return pd.DataFrame(missing_stats) if missing_stats else pd.DataFrame()

    def analise_numericas(self) -> pd.DataFrame:
        """
        Análise estatística das variáveis numéricas.

        Returns:
            DataFrame com estatísticas descritivas + outliers
        """
        print("\n" + "=" * 60)
        print("ANÁLISE DE VARIÁVEIS NUMÉRICAS")
        print("=" * 60 + "\n")

        # Estatísticas descritivas
        stats = self.df[self.numeric_cols].describe().T

        # Adicionar detecção de outliers (IQR method)
        outlier_info = []
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((self.df[col] < lower_bound) |
                        (self.df[col] > upper_bound)).sum()
            outlier_pct = (outliers / len(self.df)) * 100

            outlier_info.append({
                'variavel': col,
                'outliers': outliers,
                'outlier_pct': outlier_pct
            })

        df_outliers = pd.DataFrame(outlier_info)

        # Merge com estatísticas
        stats_extended = stats.reset_index()
        stats_extended = stats_extended.merge(
            df_outliers, left_on='index', right_on='variavel')

        print(stats_extended.to_string(index=False))

        # Alertas
        high_outliers = df_outliers[df_outliers['outlier_pct'] > 5]
        if not high_outliers.empty:
            print(
                f"\n{len(high_outliers)} variável(eis) com >5% de outliers:")
            for _, row in high_outliers.iterrows():
                print(f"   • {row['variavel']}: {row['outlier_pct']:.2f}%")

        return stats_extended

    def analise_categoricas(self) -> Dict[str, pd.DataFrame]:
        """
        Análise das variáveis categóricas.

        Returns:
            Dicionário com distribuição de frequências de cada variável
        """
        print("\n" + "=" * 60)
        print("ANÁLISE DE VARIÁVEIS CATEGÓRICAS")
        print("=" * 60 + "\n")

        distributions = {}

        for col in self.categorical_cols:
            print(f"\n{col.upper()}")
            print("-" * 40)

            freq = self.df[col].value_counts()
            freq_pct = self.df[col].value_counts(normalize=True) * 100

            dist_df = pd.DataFrame({
                'categoria': freq.index,
                'frequencia': freq.values,
                'percentual': freq_pct.values
            })

            distributions[col] = dist_df

            # Mostrar top 10 ou todas se forem poucas
            top_n = min(10, len(dist_df))
            for idx, row in dist_df.head(top_n).iterrows():
                bar = '█' * int(row['percentual'] / 2)
                print(
                    f"  {row['categoria']:.<20} {row['frequencia']:>6} ({row['percentual']:>5.2f}%) {bar}")

            if len(dist_df) > top_n:
                print(f"  ... e mais {len(dist_df) - top_n} categorias")

            # Alerta para alta cardinalidade
            if len(dist_df) > 50:
                print(
                    f"  Alta cardinalidade: {len(dist_df)} categorias únicas!")

        return distributions

    def analise_target(self, target_col: str = 'y') -> None:
        """
        Análise específica da variável target.

        Args:
            target_col: Nome da coluna target
        """
        print("\n" + "=" * 60)
        print("ANÁLISE DA VARIÁVEL TARGET")
        print("=" * 60 + "\n")

        if target_col not in self.df.columns:
            print(f"❌ Coluna '{target_col}' não encontrada!")
            return

        target_dist = self.df[target_col].value_counts()
        target_pct = self.df[target_col].value_counts(normalize=True) * 100

        print(f"Variável: '{target_col}'")
        print(f"\nDistribuição:")
        for cat, count in target_dist.items():
            pct = target_pct[cat]
            print(f"  {cat}: {count:,} ({pct:.2f}%)")

        # Calcular desbalanceamento
        if len(target_dist) == 2:
            ratio = target_dist.max() / target_dist.min()
            print(f"\nRatio de desbalanceamento: 1:{ratio:.2f}")

            if ratio > 3:
                print(" DATASET DESBALANCEADO! Considere:")
                print("   • SMOTE para oversampling da classe minoritária")
                print("   • Undersampling da classe majoritária")
                print("   • Class weights nos modelos")
                print("   • Métricas: F1-Score, AUC-ROC, Precision-Recall")

    def resumo_problemas(self) -> None:
        """Gera um resumo executivo dos principais problemas encontrados."""
        print("\n" + "=" * 60)
        print("RESUMO EXECUTIVO - PROBLEMAS DETECTADOS")
        print("=" * 60 + "\n")

        problemas = []

        # 1. Valores faltantes críticos
        for col in self.df.columns:
            unknown_pct = (
                (self.df[col] == 'unknown').sum() / len(self.df)) * 100
            null_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
            total_pct = unknown_pct + null_pct

            if total_pct > 20:
                problemas.append(
                    f"{col}: {total_pct:.1f}% valores faltantes/unknown")
            elif total_pct > 5:
                problemas.append(
                    f"{col}: {total_pct:.1f}% valores faltantes/unknown")

        # 2. Outliers significativos
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers_pct = (
                ((self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)).sum() / len(self.df)) * 100

            if outliers_pct > 10:
                problemas.append(f"{col}: {outliers_pct:.1f}% outliers")
            elif outliers_pct > 5:
                problemas.append(f"{col}: {outliers_pct:.1f}% outliers")

        # 3. Variáveis quase constantes
        for col in self.df.columns:
            top_freq_pct = (
                self.df[col].value_counts().iloc[0] / len(self.df)) * 100
            if top_freq_pct > 95:
                problemas.append(
                    f"{col}: {top_freq_pct:.1f}% de uma única categoria (quase constante)")

        if problemas:
            for problema in problemas:
                print(f"  {problema}")
        else:
            print("Nenhum problema crítico detectado!")

        # Recomendações
        print("\nRECOMENDAÇÕES DE TRATAMENTO:")
        print("  1. Valores faltantes: imputação, remoção ou criar flag 'is_missing'")
        print("  2. Outliers: transformação (log, box-cox), capping ou remoção")
        print("  3. Desbalanceamento: técnicas de balanceamento (SMOTE, undersampling)")
        print("  4. Variáveis categóricas: encoding (OneHot, Label, Target)")
        print("  5. Feature engineering: criar features de interação e agregação")

    def executar_eda_completo(self) -> None:
        """Executa todas as análises em sequência."""
        self.info_basica()
        self.analise_missings()
        self.analise_numericas()
        self.analise_categoricas()
        self.analise_target()
        self.resumo_problemas()


# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    # Instanciar a classe
    eda = BankDataEDA('bankfull.csv', delimiter=';')

    # Executar EDA completo
    eda.executar_eda_completo()

    # Se quiser executar análises individuais:
    # eda.info_basica()
    # missing_df = eda.analise_missings()
    # stats_df = eda.analise_numericas()
    # dist_dict = eda.analise_categoricas()
    # eda.analise_target('y')
