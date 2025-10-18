from datetime import date
class BankAccount():
    def __init__(self,holder_name='',acc_type='saving',acc_num='1234'):
        self.balance=0
        self.acc_type=acc_type
        self.acc_num=acc_num
        self.holder_name=holder_name
        self.transaction_hist={'row':[0],'date':[date.today().strftime("%Y%m%d")] ,'deposite':[0],'withdraw':[0],'balance':[0]}
        
    
    def depositing_Money(self,deposite):

        withdraw=0
        self.balance+=deposite-withdraw
        current_date=date.today().strftime("%Y%m%d") 
        row=self.transaction_hist['row'][-1]
        
        self.transaction_hist['row'].append(row+1)
        self.transaction_hist['date'].append(current_date)
        self.transaction_hist['deposite'].append(deposite)
        self.transaction_hist['withdraw'].append(withdraw)
        self.transaction_hist['balance'].append(self.balance)
        
        
    def withdrawing_Money(self,withdraw):
        deposite=0
        self.balance+=deposite-withdraw
        current_date=date.today().strftime("%Y%m%d") 
        row=self.transaction_hist['row'][-1]
        
        self.transaction_hist['row'].append(row+1)
        self.transaction_hist['date'].append(current_date)
        self.transaction_hist['deposite'].append(deposite)
        self.transaction_hist['withdraw'].append(withdraw)
        self.transaction_hist['balance'].append(self.balance)
        
    def Checking_balance(self):
        print(f"Your Balance is {self.balance}")
        
    def Getting_Account_type(self):
        suggested_acc='checking'
        if self.acc_type=='saving': suggested_acc='checking'
        return print(f"You have a {self.acc_type} with us. I suggest you open a {suggested_acc} account as well")
        
    def Getting_Account_number(self):
        return print(f"Your account number is {self.acc_num}.")
        
    def Getting_Holder_name():
        return print(f"Account holder name  is {self.holder_name}.")

    def transaction_history_req(self,n_transaction):
        num_entries = self.transaction_hist['row'][-1]
        if num_entries<n_transaction:
            n_transaction=num_entries
        print(f"------- List of Last {n_transaction} Transaction -----------")
        for key, values in self.transaction_hist.items():
            print(f"{key}: {values[-n_transaction:]}")
  
amin=BankAccount(holder_name='Amin K',acc_type='checking',acc_num='123456')
amin.depositing_Money(10)
amin.depositing_Money(50)
amin.withdrawing_Money(5)
amin.depositing_Money(20)
amin.Checking_balance()
amin.Getting_Account_number()
amin.transaction_history_req(2)