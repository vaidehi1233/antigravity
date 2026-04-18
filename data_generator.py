import pandas as pd
import numpy as np
import os

def generate_loan_data(num_samples=1000, seed=42):
    np.random.seed(seed)
    
    # 1. Input Features
    credit_score = np.random.randint(300, 851, num_samples)
    age = np.random.randint(18, 71, num_samples)
    monthly_income = np.random.randint(1500, 15001, num_samples)
    previous_loan = np.random.choice(['Yes', 'No'], num_samples, p=[0.4, 0.6])
    
    # Create initial DataFrame
    df = pd.DataFrame({
        'Credit_Score': credit_score,
        'Age': age,
        'Monthly_Income': monthly_income,
        'Previous_Loan': previous_loan
    })
    
    # 2. Logic for Eligibility (Classification Target)
    # Higher credit score, higher income, and older age (to some extent) increase chances
    eligibility_score = (
        (df['Credit_Score'] - 300) / 550 * 0.5 + 
        (df['Monthly_Income'] - 1500) / 13500 * 0.3 + 
        (df['Age'] - 18) / 52 * 0.1 +
        (df['Previous_Loan'].map({'Yes': 1, 'No': 0})) * 0.1
    )
    
    # Add some noise
    eligibility_score += np.random.normal(0, 0.05, num_samples)
    
    # Set threshold for eligibility
    df['Is_Eligible'] = (eligibility_score > 0.4).map({True: 'Yes', False: 'No'})
    
    # 3. Logic for Loan Terms (Regression Targets) - Only relevant for eligible users
    # But for training, we can fill them for all or just handle them during training.
    # Usually, we train regression only on approved cases.
    
    # Loan Amount: roughly 3-10 times monthly income based on credit score
    df['Loan_Amount'] = df['Monthly_Income'] * np.random.uniform(3, 10, num_samples) * (df['Credit_Score'] / 850)
    df['Loan_Amount'] = df['Loan_Amount'].round(-2) # Round to nearest hundred
    
    # Interest Rate: 5% to 25%, lower for higher credit scores
    df['Interest_Rate'] = 25 - (df['Credit_Score'] - 300) / 550 * 20 + np.random.normal(0, 1, num_samples)
    df['Interest_Rate'] = df['Interest_Rate'].clip(5, 25).round(2)
    
    # Loan Tenure: 12 to 60 months
    df['Loan_Tenure'] = np.random.choice([12, 24, 36, 48, 60], num_samples)
    
    # Save to CSV
    df.to_csv('loan_data.csv', index=False)
    print(f"Generated {num_samples} samples and saved to loan_data.csv")

if __name__ == "__main__":
    generate_loan_data(2000)
