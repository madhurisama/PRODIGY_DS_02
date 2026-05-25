import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("train.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)




print("\n── Missing Values ──")
print(df.isnull().sum())


df["Age"].fillna(df["Age"].median(), inplace=True)


df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)


df.drop(columns=["Cabin", "PassengerId", "Name", "Ticket"], inplace=True)


df["Sex"] = df["Sex"].astype("category")
df["Embarked"] = df["Embarked"].astype("category")

print("\n── After Cleaning - Missing Values ──")
print(df.isnull().sum())
print("\nCleaned Shape:", df.shape)



print("\n── Summary Statistics ──")
print(df.describe())

survival_rate = df["Survived"].mean() * 100
print(f"\nOverall Survival Rate: {survival_rate:.1f}%")


sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(18, 20))
fig.suptitle("Titanic EDA Dashboard", fontsize=22, fontweight="bold", y=1.01)



ax1 = fig.add_subplot(4, 3, 1)
sns.countplot(x="Survived", data=df, palette=["#e74c3c", "#2ecc71"], ax=ax1)
ax1.set_title("Survival Count")
ax1.set_xticklabels(["Did Not Survive", "Survived"])
ax1.set_xlabel("")
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height())}',
                 (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='bottom', fontweight='bold')



ax2 = fig.add_subplot(4, 3, 2)
sns.countplot(x="Sex", hue="Survived", data=df,
              palette=["#e74c3c", "#2ecc71"], ax=ax2)
ax2.set_title("Survival by Gender")
ax2.legend(["Did Not Survive", "Survived"])
for p in ax2.patches:
    ax2.annotate(f'{int(p.get_height())}',
                 (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='bottom', fontsize=9)



ax3 = fig.add_subplot(4, 3, 3)
sns.countplot(x="Pclass", hue="Survived", data=df,
              palette=["#e74c3c", "#2ecc71"], ax=ax3)
ax3.set_title("Survival by Passenger Class")
ax3.set_xlabel("Class (1=First, 2=Second, 3=Third)")
ax3.legend(["Did Not Survive", "Survived"])
for p in ax3.patches:
    ax3.annotate(f'{int(p.get_height())}',
                 (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='bottom', fontsize=9)



ax4 = fig.add_subplot(4, 3, 4)
ax4.hist(df[df["Survived"] == 0]["Age"], bins=30,
         alpha=0.6, color="#e74c3c", label="Did Not Survive")
ax4.hist(df[df["Survived"] == 1]["Age"], bins=30,
         alpha=0.6, color="#2ecc71", label="Survived")
ax4.set_title("Age Distribution by Survival")
ax4.set_xlabel("Age")
ax4.set_ylabel("Count")
ax4.legend()



ax5 = fig.add_subplot(4, 3, 5)
ax5.hist(df[df["Survived"] == 0]["Fare"], bins=40,
         alpha=0.6, color="#e74c3c", label="Did Not Survive")
ax5.hist(df[df["Survived"] == 1]["Fare"], bins=40,
         alpha=0.6, color="#2ecc71", label="Survived")
ax5.set_title("Fare Distribution by Survival")
ax5.set_xlabel("Fare")
ax5.set_ylabel("Count")
ax5.legend()



ax6 = fig.add_subplot(4, 3, 6)
sns.countplot(x="Embarked", hue="Survived", data=df,
              palette=["#e74c3c", "#2ecc71"], ax=ax6)
ax6.set_title("Survival by Embarked Port")
ax6.set_xlabel("Port (C=Cherbourg, Q=Queenstown, S=Southampton)")
ax6.legend(["Did Not Survive", "Survived"])



ax7 = fig.add_subplot(4, 3, 7)
gender_survival = df.groupby("Sex")["Survived"].mean() * 100
gender_survival.plot(kind="bar", color=["#3498db", "#e91e8c"],
                     edgecolor="white", ax=ax7)
ax7.set_title("Survival Rate by Gender (%)")
ax7.set_ylabel("Survival Rate (%)")
ax7.set_xlabel("")
ax7.set_xticklabels(["Female", "Male"], rotation=0)
for p in ax7.patches:
    ax7.annotate(f'{p.get_height():.1f}%',
                 (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='bottom', fontweight='bold')



ax8 = fig.add_subplot(4, 3, 8)
class_survival = df.groupby("Pclass")["Survived"].mean() * 100
class_survival.plot(kind="bar", color=["#f39c12", "#8e44ad", "#16a085"],
                    edgecolor="white", ax=ax8)
ax8.set_title("Survival Rate by Class (%)")
ax8.set_ylabel("Survival Rate (%)")
ax8.set_xlabel("Passenger Class")
ax8.set_xticklabels(["1st", "2nd", "3rd"], rotation=0)
for p in ax8.patches:
    ax8.annotate(f'{p.get_height():.1f}%',
                 (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='bottom', fontweight='bold')



ax9 = fig.add_subplot(4, 3, 9)
sns.boxplot(x="Pclass", y="Age", data=df,
            palette=["#f39c12", "#8e44ad", "#16a085"], ax=ax9)
ax9.set_title("Age Distribution by Class")
ax9.set_xlabel("Passenger Class")



ax10 = fig.add_subplot(4, 3, 10)
corr = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax10)
ax10.set_title("Correlation Heatmap")



ax11 = fig.add_subplot(4, 3, 11)
sns.countplot(x="SibSp", hue="Survived", data=df,
              palette=["#e74c3c", "#2ecc71"], ax=ax11)
ax11.set_title("Siblings/Spouses vs Survival")
ax11.set_xlabel("Number of Siblings/Spouses")
ax11.legend(["Did Not Survive", "Survived"])



ax12 = fig.add_subplot(4, 3, 12)
sns.countplot(x="Parch", hue="Survived", data=df,
              palette=["#e74c3c", "#2ecc71"], ax=ax12)
ax12.set_title("Parents/Children vs Survival")
ax12.set_xlabel("Number of Parents/Children")
ax12.legend(["Did Not Survive", "Survived"])


plt.tight_layout()
plt.savefig("titanic_eda.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nEDA Dashboard saved as titanic_eda.png")


print("\n══════════════════════════════════")
print("         KEY FINDINGS")
print("══════════════════════════════════")
print(f"Overall Survival Rate     : {df['Survived'].mean()*100:.1f}%")
print(f"Female Survival Rate      : {df[df['Sex']=='female']['Survived'].mean()*100:.1f}%")
print(f"Male Survival Rate        : {df[df['Sex']=='male']['Survived'].mean()*100:.1f}%")
print(f"1st Class Survival Rate   : {df[df['Pclass']==1]['Survived'].mean()*100:.1f}%")
print(f"2nd Class Survival Rate   : {df[df['Pclass']==2]['Survived'].mean()*100:.1f}%")
print(f"3rd Class Survival Rate   : {df[df['Pclass']==3]['Survived'].mean()*100:.1f}%")
print(f"Average Age (Survived)    : {df[df['Survived']==1]['Age'].mean():.1f}")
print(f"Average Age (Not Survived): {df[df['Survived']==0]['Age'].mean():.1f}")
print(f"Average Fare (Survived)   : ${df[df['Survived']==1]['Fare'].mean():.2f}")
print(f"Average Fare (Not Survived): ${df[df['Survived']==0]['Fare'].mean():.2f}")