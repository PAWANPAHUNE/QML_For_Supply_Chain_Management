import pandas as pd

# Load preprocessed datasets
train_df = pd.read_csv("Training_BOP_preprocessed.csv")
test_df = pd.read_csv("Testing_BOP_preprocessed.csv")

# Count occurrences of 0 and 1 in went_on_backorder
train_counts = train_df['went_on_backorder'].value_counts().sort_index()
test_counts = test_df['went_on_backorder'].value_counts().sort_index()

# Print counts
print("Training dataset - went_on_backorder counts:")
print(f"0 (Not on backorder): {train_counts.get(0, 0)}")
print(f"1 (On backorder): {train_counts.get(1, 0)}")
print("\nTesting dataset - went_on_backorder counts:")
print(f"0 (Not on backorder): {test_counts.get(0, 0)}")
print(f"1 (On backorder): {test_counts.get(1, 0)}")

# Bar chart for Training dataset

{
  "type": "bar",
  "data": {
    "labels": ["Not on Backorder (0)", "On Backorder (1)"],
    "datasets": [{
      "label": "Training Dataset",
      "data": [train_counts.get(0, 0), train_counts.get(1, 0)],
      "backgroundColor": ["#36A2EB", "#FF6384"],
      "borderColor": ["#2A8BBF", "#CC4F67"],
      "borderWidth": 1
    }]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": True,
        "title": {
          "display": True,
          "text": "Count"
        }
      },
      "x": {
        "title": {
          "display": True,
          "text": "went_on_backorder"
        }
      }
    },
    "plugins": {
      "legend": {
        "display": True,
        "position": "top"
      },
      "title": {
        "display": True,
        "text": "Class Distribution in Training Dataset"
      }
    }
  }
}


# Bar chart for Testing dataset

{
  "type": "bar",
  "data": {
    "labels": ["Not on Backorder (0)", "On Backorder (1)"],
    "datasets": [{
      "label": "Testing Dataset",
      "data": [test_counts.get(0, 0), test_counts.get(1, 0)],
      "backgroundColor": ["#36A2EB", "#FF6384"],
      "borderColor": ["#2A8BBF", "#CC4F67"],
      "borderWidth": 1
    }]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": True,
        "title": {
          "display": True,
          "text": "Count"
        }
      },
      "x": {
        "title": {
          "display": True,
          "text": "went_on_backorder"
        }
      }
    },
    "plugins": {
      "legend": {
        "display": True,
        "position": "top"
      },
      "title": {
        "display": True,
        "text": "Class Distribution in Testing Dataset"
      }
    }
  }
}
