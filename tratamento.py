import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class BankDataTreatment:
    """
    Classe para tratamento e preparação dos dados do Dataset.

    Versão 2: Inclui features de interação e ponderação temporal.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa com o DataFrame pós EDA.

        Args:
            df: DataFrame original
        """
        self.df_original = df.copy()
        self.df = df.copy()
        self.feature_engineering_log = []

    def tratar_valores_faltantes(self, strategy: str = 'smart') -> pd.DataFrame:
        """
        Trata valores faltantes e 'unknown'.

        Args:
            strategy: 'smart' (contextual), 'drop', 'mode', 'create_flag'

        Returns:
            DataFrame tratado

        Justificativa das escolhas:
        - 'poutcome' (81% unknown): representa "nunca foi contatado antes"
          → Transformar em categoria válida 'never_contacted'
          + criar flag binária
        - 'contact' (28% unknown): tipo de contato desconhecido
          → Manter como categoria 'unknown' (pode ser padrão informativo)
        - 'education' (4% unknown): imputar com moda ou criar categoria
        - 'job' (0.6% unknown): imputar com moda
        """
        print("=" * 60)
        print("TRATAMENTO DE VALORES FALTANTES")
        print("=" * 60 + "\n")

        if strategy == 'smart':
            # poutcome: 81% unknown - transformar em informação útil
            self.df['never_contacted_before'] = (
                self.df['poutcome'] == 'unknown').astype(int)
            self.df['poutcome'] = self.df['poutcome'].replace(
                'unknown', 'never_contacted')
            print(""" poutcome: 'unknown'
                  → 'never_contacted' + flag binária criada""")
            self.feature_engineering_log.append(
                "never_contacted_before: flag se cliente nunca foi contatado")

            # contact: manter 'unknown' como categoria válida
            print(
                """ contact:'unknown'
                mantido como categoria (pode indicar canal offline)""")

            # education: imputar com moda por job
            # (profissões similares têm educação similar)
            mode_by_job = self.df.groupby('job')['education'].apply(
                lambda x: x.mode()[0] if not x.mode().empty else 'secondary'
            )
            mask = self.df['education'] == 'unknown'
            self.df.loc[mask, 'education'] = self.df.loc[mask,
                                                         'job'].map(mode_by_job)
            print(
                f""" education: {mask.sum()}
                'unknown' imputados com moda por profissão""")

            # job: poucos casos, imputar com moda geral
            mode_job = self.df['job'].mode()[0]
            mask = self.df['job'] == 'unknown'
            self.df.loc[mask, 'job'] = mode_job
            print(
                f""" job: {mask.sum()}
                'unknown' imputados com moda ('{mode_job}')""")

        elif strategy == 'drop':
            # Remove todas as linhas com 'unknown'
            mask = (self.df == 'unknown').any(axis=1)
            self.df = self.df[~mask]
            print(f" Removidas {mask.sum()} linhas com 'unknown'")

        elif strategy == 'mode':
            # Imputa tudo com moda
            for col in self.df.select_dtypes(include='object').columns:
                mode_val = self.df[col].mode()[0]
                self.df[col] = self.df[col].replace('unknown', mode_val)
            print(" Todos 'unknown' substituídos por moda da coluna")

        print(f"\n Shape após tratamento: {self.df.shape}\n")
        return self.df

    def feature_engineering(self) -> pd.DataFrame:
        """
        Cria features derivadas baseadas em conhecimento de negócio.

        Features criadas:
        1. Segmentação demográfica
        2. Indicadores de comportamento financeiro
        3. Features de campanha
        4. Features temporais
        5. Features de interação 
        6. Features ponderadas temporalmente 

        Justificativa: Features derivadas capturam padrões não-lineares
        e interações que modelos lineares não detectam naturalmente.
        """
        print("=" * 60)
        print("FEATURE ENGINEERING V2")
        print("=" * 60 + "\n")

        # ===================================================================
        # FEATURES ORIGINAIS (V1)
        # ===================================================================

        # SEGMENTAÇÃO DEMOGRÁFICA
        self.df['age_group'] = pd.cut(
            self.df['age'],
            bins=[0, 25, 35, 50, 65, 100],
            labels=['jovem', 'adulto_jovem', 'meia_idade',
                    'pre_aposentado', 'aposentado']
        )
        print(" age_group: segmentação etária (5 grupos)")
        self.feature_engineering_log.append(
            "age_group: segmento demográfico por idade")

        # Perfil socioeconômico
        high_edu = ['tertiary']
        white_collar = ['management', 'admin.', 'technician']

        self.df['high_education'] = self.df['education'].isin(
            high_edu).astype(int)
        self.df['white_collar'] = self.df['job'].isin(white_collar).astype(int)
        print(" high_education: flag ensino superior")
        print(" white_collar: flag trabalho administrativo/gerencial")

        # COMPORTAMENTO FINANCEIRO
        self.df['balance_category'] = pd.cut(
            self.df['balance'],
            bins=[-np.inf, 0, 500, 2000, np.inf],
            labels=['negativo', 'baixo', 'medio', 'alto']
        )
        print(" balance_category: categorização do saldo bancário")

        # Cliente endividado (empréstimos)
        self.df['tem_dividas'] = ((self.df['loan'] == 'yes') | (
            self.df['housing'] == 'yes')).astype(int)
        self.df['endividamento_total'] = (
            (self.df['loan'] == 'yes').astype(int) +
            (self.df['housing'] == 'yes').astype(int)
        )
        print(" tem_dividas: flag se possui qualquer tipo de empréstimo")
        print(" endividamento_total: quantidade de tipos de empréstimos (0-2)")

        # Cliente inadimplente
        self.df['default_numeric'] = (self.df['default'] == 'yes').astype(int)

        # FEATURES DE CAMPANHA
        self.df['contato_intensivo'] = (self.df['campaign'] > 3).astype(int)
        print(" contato_intensivo: flag se recebeu >3 contatos na campanha")

        # Se cliente recorrente
        self.df['cliente_recorrente'] = (self.df['previous'] > 0).astype(int)

        # Tempo desde último contato
        self.df['pdays_category'] = pd.cut(
            self.df['pdays'].replace(-1, 999),
            bins=[-1, 0, 7, 30, 180, 999, 1000],
            labels=['nunca', 'esta_semana', 'este_mes',
                    'ultimos_6_meses', 'mais_6_meses', "sem_info_recente"]
        )
        print(" pdays_category: tempo desde último contato categorizado")

        # Taxa de sucesso em contatos anteriores
        self.df['previous_success'] = (
            self.df['poutcome'] == 'success').astype(int)

        # FEATURES TEMPORAIS
        month_order = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        self.df['month_number'] = self.df['month'].map(month_order)

        # Trimestre
        self.df['quarter'] = pd.cut(
            self.df['month_number'],
            bins=[0, 3, 6, 9, 12],
            labels=['Q1', 'Q2', 'Q3', 'Q4']
        )
        print(" quarter: trimestre do ano")

        # Período do mês
        self.df['periodo_mes'] = pd.cut(
            self.df['day'],
            bins=[0, 10, 20, 31],
            labels=['inicio', 'meio', 'fim']
        )
        print(" periodo_mes: período do mês do contato")

        # Score de propensão (heurística simples baseada em padrões do EDA)
        self.df['propensity_score'] = (
            (self.df['balance'] > 1000).astype(int) * 3 +
            (self.df['previous_success'] == 1).astype(int) * 4 +
            (self.df['pdays'] < 30).astype(int) * 3 +
            (self.df['age'] > 60).astype(int) * 2 +
            (self.df['campaign'] <= 2).astype(int) * 2
        )
        print(" propensity_score: score heurístico de propensão (0-12)")

        # Interação Balance × Previous Success
        self.df['high_balance_prev_success'] = (
            (self.df['balance'] > 2000) &
            (self.df['previous_success'] == 1)
        ).astype(int)
        print(" high_balance_prev_success: saldo alto + sucesso anterior")
        self.feature_engineering_log.append(
            """high_balance_prev_success: interação saldo alto
              + histórico positivo"""
        )

        #  Contact History Weighted (ponderação temporal)
        def contact_history_score(row):
            """Pondera histórico de contato pela recência."""
            if row['pdays'] == -1:  # nunca contatado
                return 0
            elif row['poutcome'] == 'success':
                # Sucesso recente vale mais (decay exponencial)
                return 5 / (1 + row['pdays']/30)
            elif row['poutcome'] == 'failure':
                return -2 / (1 + row['pdays']/30)
            else:  # other
                return -1 / (1 + row['pdays']/30)

        self.df['contact_history_weighted'] = self.df.apply(
            contact_history_score, axis=1
        )
        print(" contact_history_weighted: histórico ponderado por recência")
        self.feature_engineering_log.append(
            "contact_history_weighted: combina pdays + poutcome com decay temporal"
        )

        # Financial Health Score
        self.df['financial_health_score'] = (
            # Positivos
            (self.df['balance'] > 1000).astype(int) * 2 +
            (self.df['default'] == 'no').astype(int) * 3 +
            (self.df['tem_dividas'] == 0).astype(int) * 2 +
            # Negativos
            -(self.df['balance'] < 0).astype(int) * 3 +
            -(self.df['default'] == 'yes').astype(int) * 5
        )
        print(" financial_health_score: score consolidado de saúde financeira")
        self.feature_engineering_log.append(
            "financial_health_score: combina balance + default + dívidas"
        )

        # Campaign Density (intensidade temporal)
        self.df['campaign_density'] = self.df.apply(
            lambda row: row['campaign'] / max(row['pdays'], 1)
            if row['pdays'] > 0
            else row['campaign'] / 30,  # assume 30 dias se nunca contatado
            axis=1
        )
        print(" campaign_density: intensidade de contatos no tempo")
        self.feature_engineering_log.append(
            "campaign_density: contatos por unidade de tempo"
        )

        # Timing Score (sazonalidade combinada)
        month_performance = {
            'may': 3, 'aug': 3, 'jul': 2, 'apr': 2, 'jun': 2,
            'mar': 1, 'sep': 1, 'oct': 1, 'nov': 1, 'dec': 1,
            'jan': 0, 'feb': 0
        }

        self.df['timing_score'] = (
            self.df['month'].map(month_performance) * 2 +
            (self.df['day'] <= 10).astype(int)  # início do mês
        )
        print(" timing_score: combina mês + período do mês")
        self.feature_engineering_log.append(
            "timing_score: captura janela temporal ótima (mês × dia)"
        )

        # Contact Efficiency
        self.df['contact_efficiency'] = self.df.apply(
            lambda row: row['campaign'] / max(row['previous'], 1)
            if row['previous'] > 0
            else row['campaign'],
            axis=1
        )
        print(" contact_efficiency: eficiência vs campanhas anteriores")
        self.feature_engineering_log.append(
            "contact_efficiency: razão campaign/previous"
        )

        print(
            f"""\n Total de features criadas: {len(self.feature_engineering_log) + 15}
            - V1 (primeiras criações)): ~20 features
            - V2 (novas após análises): 6 features de interação/ponderação"""
        )
        print(f"Shape após feature engineering: {self.df.shape}\n")

        return self.df

    def tratar_outliers(self,
                        method: str = 'cap',
                        threshold: float = 1.5
                        ) -> pd.DataFrame:
        """
        Trata outliers nas variáveis numéricas.

        Args:
            method: 'cap' (capping), 'remove', 'log_transform', 'none'
            threshold: multiplicador do IQR (padrão: 1.5)

        Justificativa usada em cada caso:
        - 'balance': valores negativos são válidos (conta no vermelho)
        - 'campaign': capping para evitar distorção
        """
        print("=" * 60)
        print("TRATAMENTO DE OUTLIERS")
        print("=" * 60 + "\n")

        numeric_cols = ['age', 'balance', 'campaign']

        if method == 'cap':
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Para balance, não aplicar lower bound
                if col == 'balance':
                    lower_bound = -np.inf

                outliers_before = ((self.df[col] < lower_bound) | (
                    self.df[col] > upper_bound)).sum()

                self.df[col] = self.df[col].clip(
                    lower=lower_bound, upper=upper_bound)

                print(f"""{col}: {outliers_before}
                      outliers tratados por capping""")

        elif method == 'log_transform':
            for col in numeric_cols:
                if col == 'balance':
                    min_val = self.df[col].min()
                    if min_val < 0:
                        self.df[f'{col}_log'] = np.log1p(
                            self.df[col] - min_val + 1)
                else:
                    self.df[f'{col}_log'] = np.log1p(self.df[col])
            print("Transformação logarítmica aplicada (colunas _log criadas)")

        elif method == 'remove':
            initial_shape = self.df.shape[0]
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                if col == 'balance':
                    lower_bound = -np.inf

                self.df = self.df[
                    (self.df[col] >= lower_bound) &
                    (self.df[col] <= upper_bound)
                ]
            removed = initial_shape - self.df.shape[0]
            print(
                f"{removed} linhas removidas ({removed/initial_shape*100:.2f}%)")

        print(f"\nShape após tratamento de outliers: {self.df.shape}\n")
        return self.df

    def encoding_categoricas(self, method: str = 'target', target_col: str = 'y') -> pd.DataFrame:
        """
        Realiza encoding das variáveis categóricas.

        Args:
            method: 'target', 'onehot', 'label', 'mixed'
            target_col: nome da coluna target

        Justificativa:
        - Target Encoding: preserva relação com target,
        evita explosão dimensional
        - OneHot: explica melhor para stakeholders,
        mas aumenta dimensionalidade
        - Mixed: target para alta cardinalidade, onehot para baixa
        """

        print("ENCODING DE VARIÁVEIS CATEGÓRICAS")

        # Separar target
        target_encoded = (self.df[target_col] == 'yes').astype(int)

        categorical_cols = self.df.select_dtypes(
            include='object').columns.tolist()
        categorical_cols.remove(target_col)

        if method == 'target':
            print("Método: Target Encoding (média do target por categoria)\n")

            for col in categorical_cols:
                # Calcular média do target por categoria
                target_means = self.df.groupby(col)[target_col].apply(
                    lambda x: (x == 'yes').mean()
                )

                # Aplicar encoding
                self.df[f'{col}_encoded'] = self.df[col].map(target_means)

                print(f"""{col}:
                      {len(target_means)} categorias → {col}_encoded""")

        elif method == 'onehot':
            print("Método: One-Hot Encoding\n")

            low_card_cols = [col for col in categorical_cols
                             if self.df[col].nunique() <= 15]

            self.df = pd.get_dummies(
                self.df,
                columns=low_card_cols,
                prefix=low_card_cols,
                drop_first=True
            )

            print(
                f"""{len(low_card_cols)} colunas transformadas em
                {len([c for c in self.df.columns if any(lc in c for lc in low_card_cols)])} dummies""")

        elif method == 'mixed':
            print("Método: Mixed (Target para alta card., OneHot para baixa)\n")

            high_card_threshold = 10

            for col in categorical_cols:
                n_unique = self.df[col].nunique()

                if n_unique > high_card_threshold:
                    # Target encoding
                    target_means = self.df.groupby(col)[target_col].apply(
                        lambda x: (x == 'yes').mean()
                    )
                    self.df[f'{col}_encoded'] = self.df[col].map(target_means)
                    print(f"{col} ({n_unique} cat.): Target Encoding")
                else:
                    # OneHot
                    dummies = pd.get_dummies(
                        self.df[col], prefix=col, drop_first=True)
                    self.df = pd.concat([self.df, dummies], axis=1)
                    print(f"{col} ({n_unique} cat.): OneHot Encoding")

        # Adicionar target encoded ao DataFrame
        self.df['target'] = target_encoded

        print(f"\nShape após encoding: {self.df.shape}\n")
        return self.df

    def preparar_para_modelagem(
        self,
        target_col: str = 'target',
        scale: bool = True,
        balance: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepara dados finais para modelagem.

        Args:
            target_col: nome da coluna target
            scale: aplicar normalização
            balance: aplicar SMOTE para balanceamento

        Returns:
            X_train, y_train, X_test, y_test (ou X, y se balance=True)

        Justificativa:
        - Scaling: essencial para modelos baseados em distância
        - SMOTE: lidar com desbalanceamento sem perder informação
        """
        print("=" * 60)
        print("PREPARAÇÃO FINAL PARA MODELAGEM")
        print("=" * 60 + "\n")

        # Separar features e target
        y = self.df[target_col]

        # Remover colunas não numéricas e target
        cols_to_drop = [target_col] + \
            self.df.select_dtypes(
                include=['object', 'category']).columns.tolist()
        cols_to_drop = [c for c in cols_to_drop if c in self.df.columns]

        X = self.df.drop(columns=cols_to_drop)

        print(f"Features finais: {X.shape[1]} colunas")
        print(f"Distribuição target: {y.value_counts().to_dict()}")

        # Normalização
        if scale:
            num_cols = X.select_dtypes(include=[np.number]).columns
            X_scaled = X.copy()
            scaler = StandardScaler()
            X_scaled[num_cols] = scaler.fit_transform(X[num_cols])
            print("\nNormalização aplicada (StandardScaler)")
            X = X_scaled

        # Balanceamento com SMOTE
        if balance:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_balanced, y_balanced = smote.fit_resample(X, y)

            print(f"\n SMOTE aplicado:")
            print(f"   Antes: {y.value_counts().to_dict()}")
            print(
                f"   Depois: {pd.Series(y_balanced).value_counts().to_dict()}")

            return X_balanced, y_balanced, None, None

        return X, y, None, None

    def pipeline_completo(
        self,
        missing_strategy: str = 'smart',
        outlier_method: str = 'cap',
        encoding_method: str = 'mixed',
        balance: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Executa pipeline completo de tratamento.

        Args:
            missing_strategy: estratégia para valores faltantes
            outlier_method: método de tratamento de outliers
            encoding_method: método de encoding
            balance: aplicar balanceamento

        Returns:
            X, y prontos para modelagem
        """
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETO DE TRATAMENTO DE DADOS")
        print("=" * 70 + "\n")

        # Valores faltantes
        self.tratar_valores_faltantes(strategy=missing_strategy)

        # Feature Engineering V2 (com novas features)
        self.feature_engineering()

        # Outliers
        self.tratar_outliers(method=outlier_method)

        # Encoding
        self.encoding_categoricas(method=encoding_method)

        # Preparação final
        X, y, _, _ = self.preparar_para_modelagem(balance=balance)

        print("\n" + "=" * 70)
        print("PIPELINE CONCLUÍDO COM SUCESSO!")
        print("=" * 70 + "\n")

        print(f"Dataset final:")
        print(f"   Shape: {X.shape}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Samples: {X.shape[0]}")
        print(
            f"   Target distribution: {pd.Series(y).value_counts().to_dict()}")

        # Salvar dados tratados
        X['target'] = y
        X.to_csv('bank_treated.csv', index=False)
        print("\n Dados tratados salvos em 'bank_treated.csv'")

        return X, y
