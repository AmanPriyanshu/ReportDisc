from discord import Webhook, RequestsWebhookAdapter
from discord_webhook import DiscordWebhook, DiscordEmbed
from tensorflow import keras

class ReportDisc:
	def __init__(self, url, round_to_decimals=4, embed_reports=False):
		self.url = url
		self.round_to_decimals = round_to_decimals
		self.webhook = Webhook.from_url(self.url, adapter=RequestsWebhookAdapter())
		self.embed_reports = embed_reports

	def report_stats(self, dictionary):
		msg = {}
		for key, item in dictionary.items():
			if type(item)==float:
				msg.update({key:round(item, self.round_to_decimals)})
			else:
				msg.update({key:item})
		if not self.embed_reports:
			self.webhook.send("Reported Metric: `"+str(msg)+"`")
		else:
			webhook = DiscordWebhook(url=self.url)
			embed = DiscordEmbed(title="Reported Metrics", color='03b2f8')
			embed.set_timestamp()
			for key, item in msg.items():
				if type(item)==float:
					item = round(item, 4)
				embed.add_embed_field(name=key, value=str(item), inline=False)
			webhook.add_embed(embed)
			response = webhook.execute()

	def report(self, text):
		self.webhook.send("Reported Text: `"+text+"`")

class TorchReportDisc(ReportDisc):
	def __init__(self, url, round_to_decimals=4, embed_reports=False):
		super().__init__(url, round_to_decimals, embed_reports)
		self.epochs_completed = 0

	def report_stats(self, dictionary):
		self.epochs_completed += 1
		msg = {"epochs_completed": self.epochs_completed}
		msg.update(dictionary)
		super().report_stats(msg)

class TFReportDisc(keras.callbacks.Callback):
	def __init__(self, url, round_to_decimals=4, embed_reports=False):
		self.reporter = ReportDisc(url, round_to_decimals, embed_reports)

	def on_epoch_end(self, epoch, logs=None):
		dictionary = {"epochs_completed": epoch+1}
		dictionary.update(logs)
		self.report_stats(dictionary)

	def report_stats(self, msg):
		self.reporter.report_stats(msg)

if __name__ == '__main__':
	with open(".secrets", "r") as f:
		webhook_url = f.read()
	rd = ReportDisc(webhook_url)
	for i in range(10):
		rd.report_stats({"index": i})