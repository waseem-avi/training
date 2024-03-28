{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "c084a7e2",
   "metadata": {},
   "source": [
    "# Titanic_Case_Study"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "d2a39828",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import Necessary Liabraries for predection\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "4198c80b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import Dataset from CSV to pandas dataframe."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "57121336",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_titanic = pd.read_csv(r\"C:\\Users\\Dell\\Desktop\\Titanic_Case_Study\\train.csv\")\n",
    "data_titanic.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "4250d430",
   "metadata": {},
   "outputs": [],
   "source": [
    "#to check shape of data(in terms of rows & columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "cebe429c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 12)"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_titanic.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "951e93bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "#checking information from data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "b3d145c3",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    object \n",
      " 11  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 83.7+ KB\n"
     ]
    }
   ],
   "source": [
    "data_titanic.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "aa87c27b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#checking null values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "e213420d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_titanic.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "43eb9a19",
   "metadata": {},
   "outputs": [],
   "source": [
    "## In \"Age\" variable 177 is null values\n",
    "## & in \"Cabin\" column there are 687 null values\n",
    "## so dropping \"Cabin\" column. (50% & above Null)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "bad6766b",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_titanic = data_titanic.drop(columns=\"Cabin\", axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "30723bae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>887</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Montvila, Rev. Juozas</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>211536</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>888</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Graham, Miss. Margaret Edith</td>\n",
       "      <td>female</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>112053</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>889</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Johnston, Miss. Catherine Helen \"Carrie\"</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>W./C. 6607</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>890</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Behr, Mr. Karl Howell</td>\n",
       "      <td>male</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>111369</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>891</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Dooley, Mr. Patrick</td>\n",
       "      <td>male</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>370376</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 11 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass  \\\n",
       "0              1         0       3   \n",
       "1              2         1       1   \n",
       "2              3         1       3   \n",
       "3              4         1       1   \n",
       "4              5         0       3   \n",
       "..           ...       ...     ...   \n",
       "886          887         0       2   \n",
       "887          888         1       1   \n",
       "888          889         0       3   \n",
       "889          890         1       1   \n",
       "890          891         0       3   \n",
       "\n",
       "                                                  Name     Sex   Age  SibSp  \\\n",
       "0                              Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                               Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                             Allen, Mr. William Henry    male  35.0      0   \n",
       "..                                                 ...     ...   ...    ...   \n",
       "886                              Montvila, Rev. Juozas    male  27.0      0   \n",
       "887                       Graham, Miss. Margaret Edith  female  19.0      0   \n",
       "888           Johnston, Miss. Catherine Helen \"Carrie\"  female   NaN      1   \n",
       "889                              Behr, Mr. Karl Howell    male  26.0      0   \n",
       "890                                Dooley, Mr. Patrick    male  32.0      0   \n",
       "\n",
       "     Parch            Ticket     Fare Embarked  \n",
       "0        0         A/5 21171   7.2500        S  \n",
       "1        0          PC 17599  71.2833        C  \n",
       "2        0  STON/O2. 3101282   7.9250        S  \n",
       "3        0            113803  53.1000        S  \n",
       "4        0            373450   8.0500        S  \n",
       "..     ...               ...      ...      ...  \n",
       "886      0            211536  13.0000        S  \n",
       "887      0            112053  30.0000        S  \n",
       "888      2        W./C. 6607  23.4500        S  \n",
       "889      0            111369  30.0000        C  \n",
       "890      0            370376   7.7500        Q  \n",
       "\n",
       "[891 rows x 11 columns]"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_titanic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "cab89522",
   "metadata": {},
   "outputs": [],
   "source": [
    "## \"Cabin\" column is removed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "cc01953a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   38.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_titanic.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "0bd1c6f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "## now replacing missing values in \"age\" with mean value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "64914b21",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_titanic[\"Age\"].fillna(data_titanic[\"Age\"].mean(), inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "135bd830",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>13.002015</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  891.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   13.002015    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   22.000000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   29.699118    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   35.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_titanic.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "4668b81d",
   "metadata": {},
   "outputs": [],
   "source": [
    "##working on 3rd missing value column \"Embarked\" \n",
    "## as we know in embarked column there is no interger value so cannot go for mean, hence using mode."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "e91a600a",
   "metadata": {},
   "outputs": [],
   "source": [
    "##finding the mode value of embarked column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "4d28105d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0    S\n",
      "dtype: object\n"
     ]
    }
   ],
   "source": [
    "print(data_titanic[\"Embarked\"].mode())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "85d10453",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "S\n"
     ]
    }
   ],
   "source": [
    "print(data_titanic[\"Embarked\"].mode()[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "6d1ec2d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_titanic[\"Embarked\"].fillna(data_titanic[\"Embarked\"].mode()[0], inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "c8bff477",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId    0\n",
       "Survived       0\n",
       "Pclass         0\n",
       "Name           0\n",
       "Sex            0\n",
       "Age            0\n",
       "SibSp          0\n",
       "Parch          0\n",
       "Ticket         0\n",
       "Fare           0\n",
       "Embarked       0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_titanic.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a55e93ec",
   "metadata": {},
   "source": [
    "### Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "314c335a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>13.002015</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  891.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   13.002015    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   22.000000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   29.699118    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   35.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_titanic.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "a1b16419",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    549\n",
       "1    342\n",
       "Name: Survived, dtype: int64"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_titanic[\"Survived\"].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6e902845",
   "metadata": {},
   "source": [
    "### 1 Data Visualisation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "96fd2f70",
   "metadata": {},
   "outputs": [],
   "source": [
    "##checking for survived & non survived cases with countplot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "e6c69bf8",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.set()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "39a3d235",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Survived', ylabel='count'>"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot('Survived', data=data_titanic)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "c07747bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "##no of survivors as per gender"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "daca0a32",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Sex', ylabel='count'>"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot('Sex', data=data_titanic)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "6cfb0350",
   "metadata": {},
   "outputs": [],
   "source": [
    "##comparing data of survivors with gender"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "775362dd",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Sex', ylabel='count'>"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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YbriiiuUm5urxsZG1dbWauzYsfr888+DExgntbi4OP39739XdXW1fD6f/vznP/uHjI6npqZGZWVlevjhh+VwOLRhwwbDe7Ut68dh7DlAkjR06FB9/fXXuv766+Xz+XT55Zdrx44dLeYZM2aMysvLdeONN6qpqUmpqamKi4sLUmKczGJiYvTQQw/pjjvukNfr1cCBA3XvvfcGtGzPnj118803KykpSVarVVdccYU8Ho8OHTrULuvHYdx4DwBgwLASAMCAcgAAGFAOAAADygEAYEA5AAAMOJUVaIPPP/9c8+fPV01NjXw+n6Kjo/X444/r3HPPDXY0oE04lRU4QQ0NDRo2bJhef/11xcbGSpLy8vKUnZ2tDRs2HPHKXuBkwbAScILq6up08ODBFhdfpaSk6E9/+pOam5u1ceNG3XLLLRo9erTGjBmjzz77TNLh7zN4+OGHJR2+QVx8fLy+++67oLwG4GjYcwDaYPHixXr++efVu3dvXXrppYqLi1NSUpKcTqcmTZqkJUuWKDIyUjt27NCECRO0bt06SdKNN96oiRMnKicnR/fee69SUlKC/EqAligHoI3cbre2bt2qrVu3asOGDZIO3xhuwYIFio6O9s+3d+9evfrqq4qJidFXX32lW2+9VSkpKXr66aeDFR04Kg5IAyeouLhYn332me6++25dc801uuaaa/TII4/o97//vdxut+Lj4/X888/759+9e7fsdrskaefOnerZs6dKS0vV0NDAt6Kh0+GYA3CCevXqpb/+9a8tvvDI5XLJ7XZr+PDh+uijj/zHEjZt2qSUlBR5PB7t2rVLTz31lF5//XWdffbZysrKCtZLAI6KYSWgDTZv3qyXXnpJe/bsUUREhLp3764HH3xQDodDa9as0cKFC+Xz+WS1WpWenq5BgwZp3LhxSkxM1F133aX9+/crOTlZTz75pK6++upgvxzAj3IAABgwrAQAMKAcAAAGlAMAwIByAAAYUA4AAAPKAQBgQDkAAAwoBwCAwf8DI9KcYFCXEYUAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot('Sex', hue=\"Survived\", data=data_titanic)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "733eab11",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Pclass', ylabel='count'>"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# cheking countplot for \"Pclass\" column\n",
    "sns.countplot('Pclass', data=data_titanic)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "ecd69e13",
   "metadata": {},
   "outputs": [],
   "source": [
    "##comparing Survived (Class wise)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "748aa7a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "##many people were travelling in 3rd class(lOWER) in Titanic."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "ee46e684",
   "metadata": {},
   "outputs": [],
   "source": [
    "# now cheking countplot for \"Embarked\" column\n",
    "# checking how many people started their journey from various locations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "741d7bfa",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Embarked', ylabel='count'>"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot('Embarked', data=data_titanic)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "4c8484f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "## most of the people have started their journey from Southampton (S)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e2180290",
   "metadata": {},
   "source": [
    "#### Checking numerical attributes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "240d7df5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\anaconda3\\lib\\site-packages\\seaborn\\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "  warnings.warn(msg, FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Age', ylabel='Density'>"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYoAAAEJCAYAAACKWmBmAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAAt5klEQVR4nO3deVxU970//tfMMAwMw84sCLjgLosY96UkJgYMxppQ0xpt8NF+423a25j6vdfUGmuaxNvY1luSmzT53aS5TZtoftrUSshNEKMxVdFGqAkuaEREZXEYGLZZGGY53z+IU1E5AnIYZng9Hw8e8cznnJn3JwPzmnM+53yOTBAEAURERD2Q+7oAIiIa2hgUREQkikFBRESiGBRERCSKQUFERKIYFEREJIpBQUREooJ8XYAUmput8Hj85/KQ2FgNmposvi5jULHPwwP77B/kchmio8N6bA/IoPB4BL8KCgB+V+9AYJ+HB/bZ//HQExERiWJQEBGRKAYFERGJYlAQEZEoBgUREYliUBARkShJg6KwsBA5OTnIysrC9u3be1zv6aefxu7du73LdXV1WLVqFRYvXowf/vCHsFqtUpZJREQiJAsKo9GI/Px87NixA3v27MHOnTtRWVl50zpPPPEE9u7d2+3x5557DitXrkRRURFSU1Px2muvSVUmDSMuD2B1uHr8cXl8XSHR0CRZUJSUlGDOnDmIioqCWq1GdnY2ioqKuq1TWFiI++67Dw888ID3MafTiePHjyM7OxsAkJube9N2RP3hcLpwvMLY44/D6fJ1iURDkmRXZjc0NECr1XqXdTodysvLu63z+OOPAwDKysq8jzU3N0Oj0SAoqKs0rVYLo9EoVZlERHQbkgWFx+OBTCbzLguC0G25J7darzfbXS82VtOn9YcCrTbc1yUMusHus2C2IVwT0mO7Wq2CNkYtaQ18n4eHQOuzZEFhMBhQWlrqXTaZTNDpdLfdLiYmBu3t7XC73VAoFL3e7npNTRa/mmtFqw2HydTu6zIGlS/6bHO40G7p6Lnd5oDJ7Zbs9fk+Dw/+2Ge5XCb6BVuyMYp58+bh6NGjMJvNsNvtKC4uRmZm5m23UyqVmDFjBj766CMAwJ49e3q1HRERSUOyoNDr9Vi3bh3y8vLw0EMP4cEHH0R6ejrWrFmDkydPim777LPPYteuXcjJyUFpaSl+8pOfSFUmERHdhkwQBP85RtNLPPQ09Pmiz1ZH11lPPZk5WY8wlXQz7/N9Hh78sc8+O/RERESBgUFBRESiGBRERCSKQUFERKIYFEREJIpBQUREohgUREQkikFBRESiGBRERCSKQUFERKIYFEREJIpBQUREohgUREQkikFBRESiGBRERCSKQUFERKIYFEREJIpBQUREohgUREQkikFBRESiGBRERCSKQUFERKIYFEREJIpBQUREohgUREQkikFBRESiGBRERCSKQUFERKIYFEREJIpBQUREohgUREQkikFBRESiJA2KwsJC5OTkICsrC9u3b7+pvaKiArm5ucjOzsYzzzwDl8sFAKipqcGqVauwbNkyPPbYY6itrZWyTCIiEiFZUBiNRuTn52PHjh3Ys2cPdu7cicrKym7rrF+/Hps3b8bevXshCAJ27doFAHj55ZexZMkSFBQUICsrC/n5+VKVSUREtyFZUJSUlGDOnDmIioqCWq1GdnY2ioqKvO21tbXo6OhARkYGACA3N9fb7vF4YLFYAAB2ux0hISFSlUlERLcRJNUTNzQ0QKvVepd1Oh3Ky8t7bNdqtTAajQCAp556CitWrMA777wDp9OJnTt39um1Y2M1d1j94NNqw31dwqAb7D4LZhvCNT1/6VCrVdDGqCWtge/z8BBofZYsKDweD2QymXdZEIRuy2LtP/3pT/H8889j0aJF2Lt3L3784x/jgw8+6La+mKYmCzweYYB6Ij2tNhwmU7uvyxhUvuizzeFCu6Wj53abAya3W7LX5/s8PPhjn+VymegXbMkOPRkMBphMJu+yyWSCTqfrsb2xsRE6nQ5msxlVVVVYtGgRACA7OxsmkwnNzc1SlUpERCIkC4p58+bh6NGjMJvNsNvtKC4uRmZmprc9ISEBKpUKZWVlAICCggJkZmYiOjoaKpUKpaWlAICysjKEhYUhJiZGqlKJiEiEZIee9Ho91q1bh7y8PDidTixfvhzp6elYs2YN1q5di7S0NGzbtg2bNm2CxWJBSkoK8vLyIJPJ8Oqrr+KFF15AR0cHwsLC8Morr0hVJhER3YZMEAT/OZjfSxyjGPp80Werw4XjFcYe22dO1iNMJdl3J77Pw4Q/9tlnYxRERBQYGBRERCSKQUFERKIYFEREJIpBQUREohgUREQkikFBRESiGBRERCSKQUFERKIYFEREJIpBQUREohgUREQkikFBRESiGBRERCSKQUFERKIYFEREJIpBQUREohgUREQkikFBRESiGBRERCSKQUFERKIYFEREJIpBQUREohgUREQkikFBRESiGBRERCSKQUFERKIYFEREJIpBQUREohgUREQkikFBRESiGBRERCSqV0Hx5JNPoqSkpM9PXlhYiJycHGRlZWH79u03tVdUVCA3NxfZ2dl45pln4HK5AAANDQ34l3/5Fzz00ENYsWIFampq+vzaREQ0MHoVFPfffz9ee+01ZGdn46233kJLS8tttzEajcjPz8eOHTuwZ88e7Ny5E5WVld3WWb9+PTZv3oy9e/dCEATs2rULAPD0009j4cKF2LNnD5YtW4Zt27b1vWdERDQgehUU3/zmN/Huu+/itddeQ1NTE5YvX47169ejvLy8x21KSkowZ84cREVFQa1WIzs7G0VFRd722tpadHR0ICMjAwCQm5uLoqIimM1mnD17FitWrAAAfOtb38JPfvKT/veQiIjuSFBvV/R4PLh06RKqq6vhdrsRGxuLX/ziF7jnnnuwdu3am9ZvaGiAVqv1Lut0um7BcmO7VquF0WjElStXMGLECGzduhWlpaXQarX4+c9/3qdOxcZq+rT+UKDVhvu6hEE32H0WzDaEa0J6bFerVdDGqCWtge/z8BBofe5VUOTn52P37t1ISkrCypUr8fLLL0OpVMJms2HhwoW3DAqPxwOZTOZdFgSh23JP7S6XC2fOnMGTTz6Jn/3sZ/jzn/+MDRs24J133ul1p5qaLPB4hF6v72tabThMpnZflzGofNFnm8OFdktHz+02B0xut2Svz/d5ePDHPsvlMtEv2L0KCrPZjDfffBOTJk3q9rharcZ//ud/3nIbg8GA0tJS77LJZIJOp+vWbjKZvMuNjY3Q6XTQarUICwvDwoULAQAPPvggtmzZ0psyiYhIAr0ao3C73TeFxLW9iAULFtxym3nz5uHo0aMwm82w2+0oLi5GZmamtz0hIQEqlQplZWUAgIKCAmRmZmLkyJEwGAz47LPPAACffvopUlJS+t4zIiIaEKJ7FM8++yyMRiPKyspgNpu9j7tcLly5ckX0ifV6PdatW4e8vDw4nU4sX74c6enpWLNmDdauXYu0tDRs27YNmzZtgsViQUpKCvLy8gAAr7zyCp599ln85je/gUajwdatWwegq0RE1B8yQRB6PJh/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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(data_titanic['Age'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "6ee72fc9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\anaconda3\\lib\\site-packages\\seaborn\\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "  warnings.warn(msg, FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Fare', ylabel='Density'>"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#checking for Fare column\n",
    "sns.distplot(data_titanic['Fare'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "38db281c",
   "metadata": {},
   "source": [
    "### HeatMap to check correlation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "e24b161a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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WCIiIiIi8kjsmpg+cOBApk6dyueff05ISAjLly/XcCwRERERkX8hI1/HI73Z1QkpUaIEvXv3JiQkhJ07d1KlShUKFSqUzqmJiIiIiEhWZFcnpH///phMJt544w3ef/99atasyZ49exg5cmR65yciIiIikiVl5Dkb6c2uGtCBAwcYMmQIK1eupHXr1gwdOpTTp0+nd24iIiIiIpIF2dUJsVgsWK1W1q1bR+3atYmNjSU2Nja9cxMRERERybK0OtYDNG/enFq1apE3b14qVKhAq1ateOmll9I7NxERERERyYLsmhPSsWNHOnTokLQs74wZMwgKCkrXxEREREREsrKMXKlIb3Z1Qvbu3cv3339PTEwMNpsNq9XKxYsXWb9+fXrnJyIiIiKSJbnyEr12tbxv3740bNgQi8VCu3btCA0NpWHDhumdm4iIiIiIZEF2VUI8PT1p1aoVFy5cICAggBEjRtCsWbP0zk1EREREJMsyurnucCy7KiFeXl7cvHmTwoULs2/fPtzc3LBYLOmdm4iIiIiIZEF2dUI6duxI7969qVevHkuWLKFJkyaULVs2vXMTEREREcmyMtoSvcuWLaNx48Y888wzzJw5M8Xjhw4dolWrVjz//PO8/fbbREVFPXLb0xyOFR4ezogRIzhx4gQVK1bEarWyYMECzpw5Q8mSJR95pyIiIiIiknGEh4czatQoFi5ciKenJ23atKFatWoULVo0KWbIkCH06NGDOnXqMHz4cCZNmkTv3r0faX9pVkL69u1LSEgI7733HiaTiWHDhuHr60vp0qWTlusVEREREZGHZzAa0/1mry1btlC9enUCAwPx9fWlUaNGrFq1KlmM1WolOjoagNjYWLy9vR+57Q+shEyaNAmAmjVr0rx580fekYiIiIiIOFZUVFSqw6YCAgIICAhI2r5y5QrBwcFJ2yEhIezfvz/Zcz755BPeeOMNhg4dio+PD/PmzXvkvNLshHh4eCT7++5tERERERF5dI64WOG0adMYP358ivu7detG9+7dk7atVisGw518bDZbsu24uDg+/fRTpk6dSvny5ZkyZQoff/wxP/zwwyPlZdcSvf9zdyIiIiIiIpKxdejQgRYtWqS4/+4qCEBYWBg7d+5M2r569SohISFJ28ePH8fLy4vy5csD8PLLLzNmzJhHzivNTsiJEydo0KBB0nZ4eDgNGjRI6hmtW7fukXcsIiIiIuLKHFEJuXfY1f3UqFGDcePGcePGDXx8fFizZg2DBg1KerxgwYJcvnyZU6dOUaRIEdatW0e5cuUeOa80OyGrV69+5BcWEREREZHMITQ0lN69e9O+fXtMJhOtW7emfPnydO7cmR49elCuXDmGDRtGr169sNls5MyZk6FDhz7y/gw2m832H+afpuUeJRy1K7mHacthZ6fg0hLMGsroTIG+Zmen4LKu3dJcQmfy97Y6OwWXZjQ67CuW3KNppYeaceA0Z99pme77KPB/C9N9H49C6+yKiIiIiIhDZY5uooiIiIhIFuOIOSEZlUM7IRoS5DweNUo7OwWXtvyLLc5OwaXVrpXT2Sm4LC8PDQdypiDfOGen4NIKevzt7BRcmL73ZHSqhIiIiIiIOMHDXNE8q3HdlouIiIiIiFOoEiIiIiIi4gwufCFwVUJERERERMShVAkREREREXECV14dS5UQERERERFxKFVCREREREScwJVXx1InRERERETECTQcS0RERERExEFUCRERERERcQJXHo7lui0XERERERGnUCVERERERMQJNCdERERERETEQVQJERERERFxAlVCREREREREHCTNSsiOHTvSfPITTzzxnyYjIiIiIuIyXHh1rDQ7IWPHjgXg5s2bnD17lkqVKmE0GtmzZw/Fixdnzpw5DklSRERERESyjjQ7IdOnTwegc+fOjB8/noIFCwJw4cIFPv/88/TPTkREREQkizIYNCckTRcvXkzqgADkyZOHixcvpltSIiIiIiKSddm1OlaZMmX4+OOPee6557DZbCxbtowqVaqkd24iIiIiIlmWK18x3a5OyODBg5kxY0bSHJAaNWrQtm3bdE1MRERERESyJrs6IZ6enjzzzDMUKVKEWrVqcenSJdzddYkREREREZFHpeuEPMCKFSvo0qULQ4YMITIykjZt2rBkyZL0zk1ERERERLIguzohEydOZPbs2fj5+ZEzZ04WLVrEDz/8kN65iYiIiIhkXUZj+t8yKLsyMxqN+Pv7J22HhIRgzMCNEhERERGRjMuuiR3FihVjxowZmM1mjhw5wqxZsyhZsmR65yYiIiIikmVpTsgDfP7554SHh+Pl5UXfvn3x9/enf//+6Z2biIiIiIhkQXZVQn7++Wdef/113n///fTOR0RERETEJRgMrju9wa5OyOXLl3nx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NG/e/F/tTEREREREEtkyUCUkPDycUaNGsXDhQjw9PWnTpg3VqlWjaNGiyeKuXbvGl19++a/3l2bLPTw8kv1997aIiIiIiGQNW7ZsoXr16gQGBuLr60ujRo1YtWpVirjPPvuMbt26/ev9PdQSVwYXnjwjIiIiIvKfcsB366ioKKKiolLcHxAQQEBAQNL2lStXCA4OTtoOCQlh//79yZ7z008/Ubp0aSpUqPCv80qzE3LixAkaNGiQtB0eHk6DBg2w2WwYDAbWrVv3rxMQEREREZH0MW3aNMaPH5/i/m7dutG9e/ekbavVmqzg8L/v+/9z/Phx1qxZw9SpU7l8+fK/zivNTsjq1av/9Q5ERERERCQlR8wJ6dChAy1atEhx/91VEICwsDB27tyZtH316lVCQkKStletWsXVq1dp1aoVJpOJK1eu0LZtW2bNmvVIeaXZCcmbN+8jvaiIiIiIiDjfvcOu7qdGjRqMGzeOGzdu4OPjw5o1axg0aFDS4z169KBHjx4AnD9/nvbt2z9yBwTsvGK6iIiIiIj8xwyG9L/ZKTQ0lN69e9O+fXuaN29O06ZNKV++PJ07d+bAgQP/edMfamK6iIiIiIhkTc2aNaNZs2bJ7ps4cWKKuHz58rF+/fp/tS91QkREREREnCEDXSfE0Vy35SIiIiIi4hSqhIiIiIiIOIHNha/Bp0qIiIiIiIg4lCohIiIiIiLOoDkhIiIiIiIijqFKiIiIiIiIE9jQnBARERERERGHUCVERERERMQJbJoTIiIiIiIi4hiqhIiIiIiIOIMLV0LUCRERERERcQJdrFBERERERMRBVAkREREREXECTUwXERERERFxEIdWQgJ9zY7cndyldq2czk7BpS0b/KezU3BpzT6r7uwUXNaasAPOTsGl+fu7OTsFl/bll5WdnYJkdJoTIiIiIiIi4hiaEyIiIiIi4gSaEyIiIiIiIuIgqoSIiIiIiDiBDc0JERERERERcQhVQkREREREnEBzQkRERERERBxElRAREREREWfQdUJEREREREQcQ5UQEREREREnsLlwPcB1Wy4iIiIiIk6hSoiIiIiIiBPYNCdERERERETEMVQJERERERFxAl0nRERERERExEFUCRERERERcQIbrjsnRJ0QEREREREn0HAsERERERERB1ElRERERETECbREr4iIiIiIiIOoEiIiIiIi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Fx8fH0aOHEn9+vV58cUXsdlsNG7c2FE5ioiIiIhIFpJmJcRgMODj4wPAtm3beOqpp5LuFxERERGRR6dKyH24ubkRFRVFTEwMR44coWbNxCt/XrhwAXf3NJ8qIiIiIiKSqjR7Em+99RbNmzfHbDbTunVrQkJCWLFiBaNGjaJr166OylFEREREJMux2lx3dFGanZBnn32Wxx9/nIiICEqWLAmAn58fgwcPplq1ag5JUEREREREspYHjqkKDQ0lNDQ0abtOnTrpmpCIiIiIiCvIyHM20luaE9NFRERERET+a5pdLiIiIiLiBKqEiIiIiIiIOIgqISIiIiIiTmB14UqIOiEiIiIiIk5gc+ElejUcS0REREREHEqVEBERERERJ9DEdBEREREREQdRJURERERExAlceWK6KiEiIiIiIuJQqoSIiIiIiDiB5oSIiIiIiIg4iCohIiIiIiJOoEqIiIiIiIiIg6gSIiIiIiLiBFodS0RERERExEFUCRERERERcQLNCREREREREXEQh1ZC/L2tjtyd3CXIN87ZKbi0qDhPZ6fg0vz93Zydgsuqu2mYs1NwaRvr9HF2Ci6txNFVzk5BMjirC381ViVEREREREQcSnNCREREREScQHNCREREREREHESVEBERERERJ1AlRERERERExEFUCRERERERcQJXvmK6OiEiIiIiIk5gc8h4LIMD9vHwNBxLREREREQcSpUQEREREREn0MR0ERERERERB1ElRERERETECaxWZ2fgPKqEiIiIiIiIQ6kSIiIiIiLiBJoTIiIiIiIi4iCqhIiIiIiIOIErX6xQlRAREREREXEoVUJERERERJxAc0JEREREREQcRJUQEREREREnsDlkUojBAft4eKqEiIiIiIiIQ6kSIiIiIiLiBK68OpbdnZCYmBgiIyOx3TWDJk+ePOmSlIiIiIiIZF12dULGjx/PpEmTyJEjR9J9BoOBdevWpVtiIiIiIiJZmSuvjmVXJ2ThwoWsX78+WSdERERERETkUdjVCQkJCSFbtmzpnYuIiIiIiMuwZoJJIRcvXuTDDz/k+vXrFC5cmJEjR+Ln55cs5sqVK/Tp04dr165hNBr56KOPePLJJ9N83TQ7IePHjwcgICCAl19+mdq1a+Pm5pb0eLdu3R61PSIiIiIiksF98cUXtG3bliZNmjBhwgS+/fZbPvzww2Qx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ixcvsn79+ofa2dKdlkdOVP6dIN84Z6fg0qLitFqUMx08nXK8szhG19yLnZ2CS9tYp4+zU3BpJY5qdS9nKfpYYWenYJfPpyWk+z4GdsiY30HsmgHZt29fGjZsiMVioV27doSGhtKwYcP0zk1ERERERLIgu4ZjeXp60qpVKy5cuEBAQAAjRozIlFdMFxERERHJKKwZd/GqdGdXJcTLy4ubN29SuHBh9u3bh5ubGxaLhlaJiIiIiMjDs6sT0rFjR3r37k29evVYsmQJTZo0oWzZsumdm4iIiIhIlmWz2tL9llGlORwrPDycESNGcOLECSpWrIjVamXBggWcOXOGkiW1/KaIiIiIiDy8NCshffv2JSQkhPfeew+TycSwYcPw9fWldOnSdl8pXEREREREUrLZ0v+WUT2wEjJp0iQAatasSfPmzR2Rk4iIiIiIZGFpdkI8PDyS/X33toiIiIiIPDprBp6zkd4eakyVwWBIrzxERERERMRFpFkJOXHiBA0aNEjaDg8Pp0GDBthsNgwGA+vWrUv3BEVEREREsiJbRp60kc7S7ISsXr3aUXmIiIiIiLgUm9XZGThPmp2QvHnzOioPERERERFxEWl2QkREREREJH1YXXg4li72ISIiIiIiDqVKiIiIiIiIE7jyxHRVQkRERERExKFUCRERERERcQJdrFBERERERMRBVAkREREREXECF54SokqIiIiIiIg4liohIiIiIiJOYNOcEBEREREREcdQJURERERExAl0xXQREREREREHUSVERERERMQJNCdERERERETEQVQJERERERFxAlVCREREREREHESVEBERERERJ3DhQogqISIiIiIi4lgOrYQYjS7c3XOygh5/OzsFl/bKx1ednYJL+/LLys5OwWWd8qzp7BRcWomjq5ydgks7VvJZZ6fgsoqajjk7BbtoToiIiIiIiIiDaE6IiIiIiIgT2Fz4iunqhIiIiIiIOIFVw7FEREREREQcQ5UQEREREREncOXhWKqEiIiIiIiIQ6kSIiIiIiLiBFqiV0RERERExEFUCRERERERcQJVQkRERERERBxElRARERERESewanUsERERERERx1AlRERERETECTQnRERERERExEFUCRERERERcQJdMV1ERERERMRBVAkREREREXECq+aEiIiIiIiIOIY6ISIiIiIiTmCz2tL99m9dvHiRdu3a8eyzz9KlSxeio6PvG3v79m0aNmzItm3bHvi66oSIiIiIiEiqvvjiC9q2bcuqVasoW7Ys33777X1jBw0aRFRUlF2vq06IiIiIiIgT2Gy2dL/9GyaTiR07dtCoUSMAWrZsyapVq1KNXbFiBX5+fpQoUcKu19bEdBERERGRLCoqKirV6kRAQAABAQFpPjciIgJ/f3/c3RO7DMHBwYSHh6eIu3jxItOmTWPatGl07tzZrrzUCRERERERcQKb1Zru+5g2bRrjx49PcX+3bt3o3r170vbKlSsZNmxYspiCBQtiMBiS3XfvttVq5dNPP6Vfv354e3vbnZc6ISIiIiIiTuCIJXo7dOhAixYtUtx/bxXkueee47nnnkt2n8lkolq1algsFtzc3Lh69SohISHJYk6dOsWpU6f49NNPATh79iyfffYZgwYNonr16vfNS50QEREREZEsyp5hV/fj4eFBlSpVWLFiBc2aNWPx4sXUrl07WUzRokXZtGlT0vZrr71Gt27dqFatWpqv/VAT0yMjIx8mXERERERE7iOjT0wH6N+/P/PmzaNx48bs3LmTXr16ATB79mzGjBnzyK9rVyXkyJEj9O7dm7i4OObOncurr77K6NGjKVOmzCPvWEREREREMra8efMyffr0FPe/8sorqcanFpsauyohgwcPZsKECQQGBhIaGsqAAQPo37+/XTsQEREREZGUMsPFCtOLXZ2Q2NhYHnvssaTtmjVrkpCQkG5JiYiIiIhI1mXXcKzAwECOHj2atCTX0qVLyZ49e7omlp4O797EijmjMZsTyF2gOC+/NQhvX/+Hiom4fomx/dry/vCF+AfkAODkoW0sm/k1FosJD09vWnToQ4Gi5R3atszmzx07mTRtBiaTiSKFCvJ+z274+fomi/l1w0bmLViCwQBeXl50fftNShQrCsCS5StZueZXEuLjKVb0Md7v2Q1PDw9nNCXTerJKEG+3L4ynh5G/zkQzbOwxYmItKeK6vVGEerWCibplBuDshRj6jzgCwC8za3D1WnxS7KyF51i76YpjGpCJ7Nv5OwtmjMdkMpG/YFE6dvscn3vOPfeLuX0rkunfD+Ps6eN4eXtTq/7zNGzSBoDTJw4xe/LXxMfFYrVaaNzidZ6s29gZTcw0dm3fysxp32M2mShQ6DHe7fUxvr5+KeJsNhvjRw2lQMEivNDqztCDVb8sYt2aX0iIj6dI0RK82+tjPDw8HdmETG379m1MmzoFk8lEocKF6dWrd4rjv379OhYumA8GA15eXrzzdheKFS9OfHw83307gePHj2Gz2ShRoiRd3u2Kl5eXk1qTNVWYPJxbB45zatRkZ6eSpWXkSkV6s6sSMmDAAL744gtOnDhBlSpVmDZtGgMHDkzv3NLF7agbzP3+Mzr0Hs0n3ywnZ0g+ls/+5qFidv62hG+/6EBUxJ0vWWZzAtPHfsCLnb/ggy8X8XSLt5n1bR+HtSszuhkZycjR4+jf5yOmfj+B3GFh/Dg1+TjCc+cv8MPknxg2sB/fjxtFu5dfZMDQLwH4fctWlixbzojBA/jx27HEJySwYPFSZzQl0woM8KBvzxJ8Nuwwbbvs4OLlWLq8XjjV2LKlstN/xBE69txFx567kjog+fP6cOuWKen+jj13qQOSiqjICCaP+4KuH33FsAkLCQ7Lx/zp4+yOmTP5a7y8fRky9mc+Gz6NA7u3sHfHb9hsNiaM+JAX2rzNF6Nm07vfOOZM+Ybwi2ed0cxMITLyJhNGD+PDvoMY+8NMQsNyM3PK9ynizp89wxd9e/Hn5k3J7v/zj02sXLaAz4eMYtR3P5GQEM8vi+Y5Kv1MLzLyJqNHfUPfT/vxw8RJhIXlZsqUKclizp8/x+RJPzJw0GDGj/+WNm1eYcj/t3fncVHV++PHXzMwgIoiIILiRiZgbojXJJfr2nUJ9CakqOk3S80MRcNyIfm5IKK4XcXE0mtqZlpAXm6WO+JSahrXckEsMzEEWdxYZJiZ3x88nBgFwoQZlvfz8eDxYOZ8zsz7c86czzmf81nO4kUA7PxsBxqNhsh164lct54HBQ/YtWunKbJSI1m7P0O3fVtwGj7Q1KGIGq5clZATJ06wY8cOTp06RXx8PNHR0bi4lHyhUtUlnTtB82fa49CkJQDdX/Tn7PGvDGYPKCvNnax0fvr+EJPmfGjwuebmFoSsO0Qzl7bodDoy01Ooa93QaPmqjs6cTcS1TRuaOTcFwGfIIA7GJxjsC5VKxTvTpmBvZweAa5vWZGffRq1Ws/9QPH4vD6NB/foolUqmvz2ZF/v2MUFOqq+unW25mHyPlNQ8AGK//p0Xezs+lk5lrqDNM9aM9m3OlrVdCJ3zHI4ORXcdO7g3QKPVsS7cg4/XdOE1/5Yon2jevdrhfOK3uLR5DsemLQDoO8iP7xK+Nvi9l5Xm2s+X6N5nCEozM8xVKjp26cmZbw9SqC5g6MhJtOtUNBWiXSNH6tvYkpX5+BNtRZH/nT3Fs23caeLcHICBL/2To/H7H5tF5puvYuk/0JsXevYxeP/Iob34DPenfv0GKJVKJgXM5O/95IKtvM6ePUsbV1ecnZ0BeOmll4g/fOixsn9a4HTs7OwBaNPGlezsbNRqNe07dMDffxRKpRIzMzNaP/Mst9LlxkdFafnWGK7/+3NSo78xdSi1glanrfS/qqpc3bE++eQT/P39qftIN5nq6HZmKg3tnfSvbewcyc+7z4O8HH13q7LS2Ng15rV3Sp6OzMxcxb3bGayc+wo597IZO21F5WammkvPyKBxI3v9a4dG9uTm5pKbl6fvkuXk2Bgnx6KH4uh0OqI2buaF57uiUqlIufE7t13vMDtkIZlZWXRo15aJ4//PJHmprhwdLEkv1o3qVsYDrOuZU7eOmUGXrEb2lpw9l81H265y9bdcRr3cjCXB7Xh9+lnMzBR8n3ibqC2/YG6uYFlIB3JyC/n8PzdMkaUqKysjDbti5YqtfWPycnPIz8vRd8kqK42La3tOxO/hWfdOFKrVnPn2IGbm5qgsLPn7gH/q14nfF0N+Xg6tXTsYLW/VTeatdOwd/njYln0jB3Jzc8jLyzXoEjThrRkA/O/saYP1U29c587tbELnzSQrK4O27Toy9vW3jBN8DXDr1i0cGjnoXzdq5EBubq7B9nd0dMLRsehY0Ol0fPTRBrp180KlUuHp2UW/bnpaGrt3xzJ1aqBxM1GDnQ8sanFq9GIPE0ciarpyVUKcnJwYN24cnTp1MuhzGRAQUGmBVRadTgePPG4eQFHs1m150pSmfsNG/L8PDpNy9QJRi9/AqVlrHJq0eqqYa6rStrOyhO2cl59PxKo1pGdkEr4gBABNoYYzP/yPhfPmYKFSsWzVGjZv3c6USW9Ueuw1hUKhoKQpxB99gmtqWj7vLvhJ/3pHbAqv+bekiaMVcftu/pHwAezcnYKfj7NUQh5R9Ht//H2l0qxcafzHz2Dnx6tY8M4YGtja086jG1cunTNI91X0Zg58tYMZ8yKxsLSq6CzUGFqdTj/GsbiSyp6SFGoKOffD98wKCUOlsiByZRg7tn7E+EnTKjrUGklX6vY3e+y9/Px8Vq5cTsatDBYuCjVYlpyczOLQhXj7DOX5P3komhBVlYwJ+RMeHh48//zzNWLQl619E4OxHHey0qlTrwGWVnWfKM2j8nLv8ePpA/rXzVyeo2kLN1J/u1zBOag5Gjs0IjMrS/86IzOT+tbW1LEyvHhKS79F4Mw5KJVmrAhbiLV10Z0ye3tbenb3ol7duqhUKvr37c2FS0lGzUN19MaYVmz+Vxc2/6sLPv9wopHdH4NpG9lbcveemvwHhs23rVvVY2DfxgbvKYDCQi0D+zamdat6Bu9rCmtvoVoa+0ZO3M7K0L/OzrxFPesGWFrVKVeavNz7vDIukEVrdvHugvXodNC4SVF3IrW6gKgVczl5bC/B4R/TwsXVeBmrhhwcHMnK/GM7Z2VmYG1dH6ti+6IsdnaN6Nb979StWw+VSsXf+/6DpIvnKyvcGsfBwYHMrEz968yMDKytrbF6pOxPT09nZtAMzJRmLAlfirX1H5M4HDkSz/vBc3jttdcZOdLfaLELISpOuSohAQEBBn9vv/02w4YNq+zYKoVrx+5cSz7HrdRrAHx7YCft/9bvidM8SqlUsnPDPK4mnQXg5vUrpP/+i8yOVYYunT24mHSZlBu/AxC3Zy/dvZ43SJObm0fQnHn07O7F+7OCDCrCvXp058jR4zx48ACdTsfxb0/qZ80Spdu0/Vf9APJJM3+gnVsDmjUpuvj65+CmHD2Z+dg6Wq2O6ZOepYlj0UXCy0OacuXXHG5lFvBMi3q8MaYVSiVYWCjx9Xbm4FHpn/2odh5e/HL5R/2A8fi9X+DxfO9yp4nfG82XO6IAuHM7k4QDX+LVaxAAH616n/y8+wQv2Uyjxk2NlaVqq5NnV5KTLpB64zoA+/bspqtXz3Kv79WjDyeOHdaXPae+O8qzru6VFW6N4+nZhaRLl7hxo6i1dM+er/DyesEgTW5uLrNnv0f37j2YNXuOQdl/8uR3bIhaT2hoGH369jVq7EJUtNr8nBCFrhzPc9+5cydLly4lLy9P/16zZs3Yv3//E33Zf88WPnmEleDiDwl89dkqNIWF2Ds2Z/SUMDLTUtj1UQhB4TGlpnl0oHnQqHYs2HBMP0XvzxdOE7c9Ao2mEHNzC4b4T6dNey9jZ69EnepXzRaZk6fPsGnLJxQWqmnSxIlZ7wSSejONlWvWsWHtKj7dFc3Hn3yKS8sWBustW7wA63r12L7zC+KPHkOr1dKm9TNMD3jrsSl+q4JR79wydQil8upix+T/c8HcXMGNm/mErrzEvfuFuD1rzeypbowPPAPAP/o05lW/FiiVcCujgPC1SaTdeoClpZJ33nyW59waYG6u4PCxDD7cdtXEuTK0dGmXP09kBOfOHOOLTyLRqNU4ODVjQuBCbqXd4ON1i1iwakepaazr25CXl8PG1fNIT01Bh46Xho/nhT5DuHLpHGFzxuPYtCUWFn9cqL0ybirtO3c3VVb1GljkmDqEEp09/S3bt3xIoVqNYxNnpgYFk3bzd6L+tYzlkYZTkkauDKN5Sxf9FL0ajYbonVs5kXAIrVaLS2tX3pw6s8Qpfk2tjiLX1CGU6PTpU0VT9BYW0sSpCUEz3+Vmair/WrOayMgP2LXzM7Zt20rLVq0M1gsLC2dm0Azu3buPfbExhc+1fY4pb1e9LuJJ7oNMHcJf1nHTEu7/lFxtp+h9SV09ekb8c0rlX599+UHVbB0vVyWkX79+bNmyhdWrVzNjxgyOHDnC2bNnWbHiyQZeV5VKSG1UVSshtUVVroTUBlWlElIbVdVKSG1RVSshtUV1roRUd9WlEjLsrcqPc/d6t0r/jr+iXN2x7O3tad68OW5ubly+fJkxY8aQlFQ9dq4QQgghhBCiainX7Fh16tThu+++w83NjQMHDtChQwfy8/MrOzYhhBBCCCFqLK226j7Ho7KV2RKSllb0sKt58+Zx+PBhevXqxe3btxk8eDCvvvqqUQIUQgghhBBC1CxltoRMnjyZ2NhY2rRpg6OjI0qlkrVr1xorNiGEEEIIIWqsqjx7VWUrsyWk+Jj1uLi4Sg9GCCGEEEIIUfOV2RJS/Imm5ZhESwghhBBCCFFOOl3tHRNSroHpYFghEUIIIYQQQjyd2twdq8xKSHJyMv379weKBqk//F+n06FQKDh48GDlRyiEEEIIIYSoUcqshOzdu9dYcQghhBBCCFGrSEtIKZydnY0VhxBCCCGEEKKWKPeYECGEEEIIIUTF0dbigellTtErhBBCCCGEEBVNWkKEEEIIIYQwgdo8JkRaQoQQQgghhBBGJS0hQgghhBBCmIBOK2NChBBCCCGEEMIopCVECCGEEEIIE5AxIUIIIYQQQghhJNISIoQQQgghhAno5DkhQgghhBBCCGEc0hIihBBCCCGECWhlTIgQQgghhBBCGIe0hAghhBBCCGEC8pwQIYQQQgghhDASaQkRQgghhBDCBOQ5IUIIIYQQQghhJNISIoQQQgghhAnU5ueESCVECCGEEEIIE5DuWEIIIYQQQghhJNISIoQQQgghhAnIFL1CCCGEEEIIYSQKnU5XezujCSGEEEIIIYxOWkKEEEIIIYQQRiWVECGEEEIIIYRRSSVECCGEEEIIYVRSCRFCCCGEEEIYlVRChBBCCCGEEEYllRAhhBBCCCGEUUklRAghhBBCCGFUUgkRQgghhBBCGJVUQoQQQgghhBBGZW7qAP6KlJQUBg0aROvWrVEoFKjVaho3bsySJUtwcnIydXhPbO3atQBMnTrV4P2UlBTGjRvHoUOHTBHWU/nmm2/48MMPKSwsRKfTMWzYMCZMmPBUn7ljxw4ARo0a9VSfM3bsWAICAujWrdtTfU5186THTUxMDKdOnSI8PNwE0dY+ly9fxsfHhzVr1jBw4EBTh1MjlVYuTZw4kdDQUI4fP17qb/7kyZOsXLmSvLw8NBoNvXv3JigoCDMzMxPkpPqprPN2aedPUbri+6K4qKgomjRpYqKoRG1ULSshAI0bN2b37t361+Hh4SxbtoyVK1eaMCoBkJaWxtKlS4mJicHW1pacnBzGjh2Li4sL/fv3/8uf+7SVDyHHTVUWHR3NoEGD2Llzp1RCKkFZ5dJHH31U5roFBQUEBQWxY8cOmjdvTkFBAdOmTWP79u2MGzfOSDmo/qT8qToe3RdCmEK1rYQ8qlu3bqxcuZKvv/6azZs3k5+fT0FBAWFhYXh6erJ582ZiY2NRKpV07NiRhQsXcunSJUJCQigsLMTS0pIlS5bQqlUrEhISWLNmDYWFhTRr1oxFixZha2tLv379GDp0KMeOHSMvL4+lS5fSvn17Ll++zOzZs9FoNPztb38jISGB/fv3k5GRQUhICDdv3kShUBAUFET37t1Zu3YtiYmJpKam8uqrrxrk48KFCwQHBwPg7u5uik351LKzs1Gr1eTn5wNQr149wsPDsbS0pF+/fmzdupVmzZpx8uRJIiMj2bZtG2PHjsXGxobk5GR8fHzIzs5m3rx5QNGJysnJiXv37gFgY2PDtWvXHlv+yiuvsHDhQpKTk9FoNEycOBFvb28KCgoIDg7mp59+wtnZmezsbNNsmCro4XFz4sQJwsPD0el0NG3alBUrVhikq4jjSpROrVYTFxfH9u3b8ff357fffqNFixacPHmS0NBQzMzM8PDw4Oeff2bbtm1cu3aN+fPnc/v2baysrJg3bx7PPfecqbNRpZWnXAK4du0aY8aM4c6dO/Tp04egoCDy8vK4f/8+eXl5AFhYWBAcHExOTg5Q1Lrq7u7O999/z4MHD5g7dy49e/Y0TUarkT87bxc/L6xevZorV66wfv16FAoFHTp0YNGiRQCcO3cOf39/0tLSGD58uLSK/EWXL19m0aJF5ObmkpWVxaRJkxg1atRj1yw9evSQ8kdUiBoxJkStVrN37148PDz47LPPiIqK4j//+Q8TJkzgww8/RKPRsGHDBqKjo4mJiUGtVpOWlsaWLVsYP348MTExjBgxgsTERLKyslixYgWbNm3iyy+/pGfPnixfvlz/XQ0bNuSLL77A39+fDRs2ADB79mwCAwPZvXs3zZs3R6PRALB48WJ8fX2JiYlh/fr1hISEcP/+faDoztqePXsYPXq0QV5mzZrFzJkziY2NpVmzZkbaghXL3d2d/v37M2DAAPz8/IiIiECr1dKyZcsy13Nzc2Pv3r2MHj2a/fv3o9Fo0Ol07Nu3j5deekmfztvbu8Tl69evp127dsTExLB9+3aioqK4fv0627ZtA4oupN9//31+++23Ss1/dfHwuGnfvj0zZ85k6dKlxMXF4erqSmxsrD6dVqt96uNKlO3IkSM0bdoUFxcXBgwYwM6dO1Gr1bz33ntERETw5ZdfYm7+xz2jWbNm8e677xIbG8uiRYuYMWOGCaOvHspbLqWkpLB27VpiYmI4c+YMBw8exMbGhjfffJPhw4fj4+NDaGgoaWlpBjeK7t+/T2xsLCtWrGD27NkUFBQYO4vVyp+dtx96eF6ws7NjyZIl/Pvf/+arr75Co9Fw5MgRADIzM9m6dSvR0dFs2rRJf54VpUtPT2fYsGH6v40bN/L5558zZcoUoqOj2bp1K8uWLdOnL37NIuWPqCjVtiXk4QEERQdHx44dCQoKwtzcnEOHDnH16lVOnTqFUqnEzMyMzp074+fnR//+/Rk/fjyOjo707t2bhQsXcvToUfr160ffvn1JSEggNTVV38Su1WqxsbHRf2+vXr0AaNOmDfv27eP27dvcuHGD3r17A+Dr66u/o3bixAl++eUX1qxZA0BhYSHXr18HoGPHjo/lKSsri/T0dHr06AHA8OHDiY6OrozNV+kWLFjAlClTOHbsGMeOHWPEiBEGlbmSPNwmdnZ2uLu7c/LkSVQqFS4uLjg4OOjTlbb8xIkT5Ofn67dZbm4uycnJnDp1ipEjRwLQqlUrOnfuXEm5rvpKOm5Gjx7NpUuXaNu2LQBBQUFA0ZgQAKVSybp1657quBJli46OxtvbG4AhQ4Ywc+ZMBg4ciL29vf5C18/Pj8WLF5OTk8NPP/3EnDlz9Ovn5uaSnZ2Nra2tSeKvLspTLvXr1w87OzsABg8ezKlTpxgwYABvvfUWI0eO5MSJExw/fpyJEycSGBjIa6+9BsCIESMAaNu2LQ4ODiQlJdGhQwej5q+qe5Lz9kMPzws//PADnp6e+vEjERERAFy8eJFevXphYWGBnZ0dtra23LlzB2trayPnrnopqTuWRqPh6NGjbNiwgcuXL5Obm6tf9nA/SPkjKlK1rYSUdADl5OTg6+vL0KFD6dq1K25ubmzfvh2ADz74gMTERBISEpgwYQLLly9n0KBBdO7cmcOHD/Pxxx8THx9Pnz598PT0JCoqCoAHDx7om9wBLC0tAVAoFACYmZmh0+lKjFGr1bJlyxYaNmwIFBXA9vb2HDhwACsrq8fSKxQKg8+qrgMe4+Pjyc3NZciQIfj6+uLr68uuXbv44osvAPR5LCwsNFiv+DYZNmwYe/bsQaVS4ePj89h3lLRcq9USERFBu3btAMjIyMDGxoZdu3YZbNfid5Rrm5KOm0uXLul/zwD37t0z+M3n5OTg5+f3VMdVaGiocTJYDWVmZnL06FHOnz/P1q1b0el03L17l4SEBLRa7WPptVotFhYWBvvx5s2b+nJGlOzPyqWHipcPWq0Wc3NzEhMTOX/+PGPGjMHb21v/FxYWpq+EFC+vH64nDD3peRv+OC+Ym5sblFNZWVn6/4tv60fPo6L8pk+fToMGDejbty9Dhgzhv//9r37Zw/0g5Y+oSDWiO9ZDv/76KwqFgsmTJ9OtWzd9l52srCyGDBmCq6srgYGB9OjRg6SkJKZPn86PP/6Iv78/gYGBXLhwgU6dOpGYmMjVq1eBoous4k2Sj6pfvz7NmzfXNwvHxcXpl3l5efHpp58CcOXKFXx8fPR9iktia2tL06ZNiY+PBzAoAKoTKysrVqxYQUpKClBU6bh48SJt27bF1taWK1euAHDw4MFSP6N///6cPn2a48eP8+KLL5ZruZeXl34GrfT0dIYOHUpqaiovvPACcXFxaLVabty4wdmzZys6y9Wai4sLmZmZ+v2yceNG/XaEijmuROl2796Nl5cXCQkJHDp0iMOHDzN58mSOHTvG3bt3SUpKAv4oW+rXr0+rVq30FwHHjx9nzJgxJou/uiirXCruyJEj3L17lwcPHrBnzx66d++OjY0NkZGRXLp0SZ/u/PnzBuvu2bMHgB9//JG7d+/i6upqhFxVf6WVL4/q0KEDiYmJ3Lp1C4CwsLAyzyHiyR0/fpxp06YxYMAAEhISAB7bF1L+iIpUo27VuLu707ZtWwYPHoxCoaBnz56cOXMGOzs7Ro4ciZ+fH3Xq1MHFxQVfX1+6du1KcHAw69atQ6VSMX/+fBwcHAgLC2P69OlotVocHR31zb6lWbZsGXPnzmX16tW4ubnp7xi8//77hISE6O/UL1u27E+biCMiIpgzZw6rV6/Gw8OjQraLsXl5eREQEMDkyZNRq9VAUTe2t99+G09PTxYtWkRkZGSZAzetrKzw9PSkoKCAevXqlWt5QEAA8+fPx9vbG41Gw7vvvkuLFi0YPXo0ycnJDB48GGdnZ7k4eISlpSURERG89957qNVqWrRowbJly9i7dy9QMceVKF1sbOxjfarHjBnDxo0b2bRpE7NmzUKpVOLi4qIvWyIiIpg/fz4bN25EpVKxatUqg7vE4nFllUvFbx4988wzTJo0ibt37+Lt7a0vp8LDw5k7dy73799HoVDQsWNHQkJC9Otdv36dl19+GYBVq1ZV25ZsYyutfHmUo6MjwcHBvPHGG2i1Wjw8PBg+fDgffPCBCaKumaZOncro0aOxtLTE3d0dZ2dnfaW9OCl/REVR6KTd8qlFRkYyYsQIGjduzL59+4iLi9PPXS6EEH+FVqtl+fLlBAQEULduXTZv3kxaWhqzZ882dWjiEbX12UNCCPE0alRLiKk0bdqU119/HXNzcxo0aMDixYtNHZIQoppTKpU0bNgQPz8/VCoVzs7OUrYIIYSoMaQlRAghhBBCCGFUNWpguhBCCCGEEKLqk0qIEEIIIYQQwqikEiKEEEIIIYQwKqmECCGEEEIIIYxKKiFCCCGEEEIIo5JKiBBCCCGEEMKo/j8QOjjpSrM6FQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1080x648 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "corr = data_titanic.corr()\n",
    "plt.figure(figsize=(15, 9))\n",
    "sns.heatmap(corr, annot=True, cmap='coolwarm')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "689ee4ed",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Embarked  \n",
       "0      0         A/5 21171   7.2500        S  \n",
       "1      0          PC 17599  71.2833        C  \n",
       "2      0  STON/O2. 3101282   7.9250        S  \n",
       "3      0            113803  53.1000        S  \n",
       "4      0            373450   8.0500        S  "
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_titanic.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "b1cc0a73",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch     Fare Embarked\n",
       "0            1         0       3    male  22.0      1      0   7.2500        S\n",
       "1            2         1       1  female  38.0      1      0  71.2833        C\n",
       "2            3         1       3  female  26.0      0      0   7.9250        S\n",
       "3            4         1       1  female  35.0      1      0  53.1000        S\n",
       "4            5         0       3    male  35.0      0      0   8.0500        S"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## drop unnecessary columns\n",
    "data_titanic = data_titanic.drop(columns=['Name', 'Ticket'], axis=1)\n",
    "data_titanic.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0d8e4510",
   "metadata": {},
   "source": [
    "### Encoding Label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "22363f55",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Categorical to Numerical for further modelling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "d8dba5e9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "male      577\n",
       "female    314\n",
       "Name: Sex, dtype: int64"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_titanic[\"Sex\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "d8a79979",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "S    646\n",
       "C    168\n",
       "Q     77\n",
       "Name: Embarked, dtype: int64"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_titanic['Embarked'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "fd166ae5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  Sex   Age  SibSp  Parch     Fare  Embarked\n",
       "0            1         0       3    1  22.0      1      0   7.2500         2\n",
       "1            2         1       1    0  38.0      1      0  71.2833         0\n",
       "2            3         1       3    0  26.0      0      0   7.9250         2\n",
       "3            4         1       1    0  35.0      1      0  53.1000         2\n",
       "4            5         0       3    1  35.0      0      0   8.0500         2"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "cols = ['Sex', 'Embarked']\n",
    "le = LabelEncoder()\n",
    "\n",
    "for col in cols:\n",
    "    data_titanic[col] = le.fit_transform(data_titanic[col])\n",
    "data_titanic.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d17102c",
   "metadata": {},
   "source": [
    "### Train_Test_Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "5bab4748",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = data_titanic.drop(columns = ['PassengerId','Survived'],axis=1)\n",
    "Y = data_titanic['Survived']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "e6d7d141",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "     Pclass  Sex        Age  SibSp  Parch     Fare  Embarked\n",
      "0         3    1  22.000000      1      0   7.2500         2\n",
      "1         1    0  38.000000      1      0  71.2833         0\n",
      "2         3    0  26.000000      0      0   7.9250         2\n",
      "3         1    0  35.000000      1      0  53.1000         2\n",
      "4         3    1  35.000000      0      0   8.0500         2\n",
      "..      ...  ...        ...    ...    ...      ...       ...\n",
      "886       2    1  27.000000      0      0  13.0000         2\n",
      "887       1    0  19.000000      0      0  30.0000         2\n",
      "888       3    0  29.699118      1      2  23.4500         2\n",
      "889       1    1  26.000000      0      0  30.0000         0\n",
      "890       3    1  32.000000      0      0   7.7500         1\n",
      "\n",
      "[891 rows x 7 columns]\n"
     ]
    }
   ],
   "source": [
    "print(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "995aec63",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0      0\n",
      "1      1\n",
      "2      1\n",
      "3      1\n",
      "4      0\n",
      "      ..\n",
      "886    0\n",
      "887    1\n",
      "888    0\n",
      "889    1\n",
      "890    0\n",
      "Name: Survived, Length: 891, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "58665f6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "##Splitting the data into training data & Test data.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "1e64b376",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "id": "56b771f8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(891, 7) (712, 7) (179, 7)\n"
     ]
    }
   ],
   "source": [
    "print(X.shape, X_train.shape, X_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a33a61ec",
   "metadata": {},
   "source": [
    "### Model Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "bc40d8d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "b6e948a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "709edd83",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 9 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Sex          891 non-null    int32  \n",
      " 4   Age          891 non-null    float64\n",
      " 5   SibSp        891 non-null    int64  \n",
      " 6   Parch        891 non-null    int64  \n",
      " 7   Fare         891 non-null    float64\n",
      " 8   Embarked     891 non-null    int32  \n",
      "dtypes: float64(2), int32(2), int64(5)\n",
      "memory usage: 55.8 KB\n"
     ]
    }
   ],
   "source": [
    "data_titanic.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "18d2bd84",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId    int64\n",
       "Survived       int64\n",
       "Pclass         int64\n",
       "Sex            int32\n",
       "Age            int32\n",
       "SibSp          int64\n",
       "Parch          int64\n",
       "Fare           int32\n",
       "Embarked       int32\n",
       "dtype: object"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_titanic.astype({'Age':'int','Fare':'int'}).dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "id": "888ebb85",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 106,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#training the Logistic Regression model with training data\n",
    "model.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "5e0e89dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "#accuracy on training data\n",
    "X_train_prediction = model.predict(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "3555c745",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1 0 0 0 0 0 1 0 0 0 1 0 0 1 0 1 0 0 0 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 1\n",
      " 0 0 0 0 0 0 1 1 0 0 1 0 1 0 1 0 0 0 0 0 0 1 0 1 0 0 1 1 0 0 1 1 0 1 0 0 1\n",
      " 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 1 0 0 1 0 0 0 1 1 1 0 1 0 0 0 0 0 1 0 0 0\n",
      " 1 1 0 0 1 0 0 1 0 0 1 0 0 1 0 1 0 1 0 1 0 1 1 1 1 1 1 0 0 1 1 1 0 0 1 0 0\n",
      " 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 1 0 1 1 1\n",
      " 0 0 0 1 0 0 0 1 0 0 1 0 0 1 1 1 0 1 0 0 0 0 0 1 1 0 1 1 1 1 0 0 0 0 0 0 0\n",
      " 0 1 0 0 1 1 1 0 0 1 0 1 1 1 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 1 0 1 0 1 0 0 0\n",
      " 0 0 0 0 0 0 1 0 1 0 0 1 0 0 1 0 0 0 1 1 0 0 0 0 1 0 1 0 0 1 0 0 0 1 0 0 0\n",
      " 0 1 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 1 1 0 0 0 1 0 1 0 0 0 0 0 0 1 1 0 1 1\n",
      " 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 1 0 0 0 0 1 1 0 0 0 1 0 1 1 1 0 0\n",
      " 0 0 1 0 0 0 1 1 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 1 1 1 0 1 1 0 0 0\n",
      " 0 1 0 1 0 0 1 1 0 0 0 0 1 0 0 0 0 1 1 0 1 0 1 0 0 0 0 0 1 0 0 0 0 1 1 0 0\n",
      " 1 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 1 1 0 1 0 0 1 0 0 0 1 1 0 1 0\n",
      " 0 0 0 0 1 0 0 1 0 1 1 0 0 1 0 0 1 0 0 0 1 0 1 1 0 0 1 1 0 1 0 1 1 1 0 1 0\n",
      " 0 1 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 1 1 1 0 0 1 1 0 0 0 0 0\n",
      " 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0\n",
      " 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 1 1 0 0 0 1 0 1 0 0 0 1 1 1 0 0 1 1\n",
      " 0 0 0 1 0 1 0 0 0 0 0 1 1 0 1 1 1 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 0\n",
      " 1 0 0 1 0 1 0 0 0 1 1 1 1 1 0 0 1 1 0 1 1 1 1 0 0 0 1 1 0 0 1 0 0 0 0 0 0\n",
      " 0 0 0 1 1 0 0 1 0]\n"
     ]
    }
   ],
   "source": [
    "print(X_train_prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "669686d1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy_score_of_training_data :  0.8132022471910112\n"
     ]
    }
   ],
   "source": [
    "training_data_accuracy = accuracy_score(Y_train, X_train_prediction)\n",
    "print('Accuracy_score_of_training_data : ', training_data_accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "c795d914",
   "metadata": {},
   "outputs": [],
   "source": [
    "# accuracy on test data\n",
    "X_test_prediction = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "1f711ed1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 1 0 1 1 0 1 0 1 1 0 0 0 0 0 0 0 0 1 1\n",
      " 0 0 0 0 0 1 0 0 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 1 0 0 0 1 0 1 0\n",
      " 1 0 0 0 1 0 1 0 0 0 1 1 0 0 1 0 0 0 0 0 0 1 0 1 0 1 1 0 1 1 0 1 1 0 0 0 0\n",
      " 0 0 0 1 1 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 1 1 1 1 0 1 0 0\n",
      " 0 1 0 0 0 0 1 0 0 1 1 0 1 0 0 0 1 1 0 0 1 0 0 1 1 1 0 0 0 0 0]\n"
     ]
    }
   ],
   "source": [
    "print(X_test_prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "id": "4954f324",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy_score_of_test data :  0.7877094972067039\n"
     ]
    }
   ],
   "source": [
    "test_data_accuracy = accuracy_score(Y_test, X_test_prediction)\n",
    "print('Accuracy_score_of_test data : ', test_data_accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6c31bb01",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
