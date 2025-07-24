class DataCollector:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.data = []

    def collect_data(self, game_data):
        self.data.append(game_data)

    def save_raw_data(self):
        with open(self.raw_data_path, 'a') as f:
            for entry in self.data:
                f.write(f"{entry}\n")
        self.data = []

    def format_data(self):
        formatted_data = []
        for entry in self.data:
            formatted_entry = self._format_entry(entry)
            formatted_data.append(formatted_entry)
        return formatted_data

    def save_processed_data(self):
        formatted_data = self.format_data()
        with open(self.processed_data_path, 'w') as f:
            json.dump(formatted_data, f)

    def _format_entry(self, entry):
        # Implement the formatting logic for each entry
        return entry  # Placeholder for actual formatting logic