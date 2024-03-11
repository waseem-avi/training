# 11.How many people have American Express as their Credit Card Provider and made a purchase above $95 ?
from dataframe import df
american_express_card_holders = df[df['CC Provider']=='American Express']
purchaser_above_95 = american_express_card_holders[american_express_card_holders['Purchase Price'] > 95]
print(purchaser_above_95.shape[0])
