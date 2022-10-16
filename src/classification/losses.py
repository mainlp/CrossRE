import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import classification_report

load_dotenv()

#
# Loss Functions
#


class LabelLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self._xe_loss = nn.CrossEntropyLoss(ignore_index=-1)


	def __repr__(self):
		return f'<{self.__class__.__name__}: loss=XEnt>'


	def forward(self, logits, targets):

		target_labels = torch.LongTensor(targets).to(logits.device)

		loss = self._xe_loss(logits, target_labels)

		return loss


	def get_classification_report(self, pred_labels, targets):

		evaluation_metrics = classification_report(targets.tolist(), pred_labels.tolist(), output_dict=True, zero_division=0)

		return evaluation_metrics

