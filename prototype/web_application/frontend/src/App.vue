<template>
  <div class="main-container">

    <div class="recommendation-panel">
      <h1 class="title">Real Time Command Recommendation</h1>
      <div class="recommendation-content">
        <div class="recommendation-header">
          <p class="subtitle">Top 5 recommended next commands</p>
          <div class="badge-count">{{ rankedItems.length }}</div>
        </div>
        <div class="recommendation-list">
          <transition-group name="list" tag="div">
            <div v-for="(item, index) in rankedItems" :key="item.name" 
                class="recommendation-item" :class="{'top-item': index === 0}">
              <div class="item-rank">{{ index + 1 }}</div>
              <div class="item-name">{{ item.name }}</div>
              <div class="item-score">{{ (item.score)}}</div>
            </div>
          </transition-group>
        </div>
      </div>
    </div>
    
    <div class="historical-commands-container">
      <div class="history-header">
        <h2 class="historical-commands-title">Historical command sequence input</h2>
        <p class="command-count">Input length: {{ commandHistory.length }}</p>
      </div>
      <div class="historical-commands-list" ref="commandList">
        <transition-group name="message" tag="div">
          <div v-for="(command, index) in commandHistory" :key="index" class="command-card">
            <div class="command-header">
              <span class="command-name">{{ command.message }}</span>
              <span class="command-cat" :class="'cat-' + command.cat">{{ command.cat }}</span>
            </div>
            <div class="command-details">
              <div class="detail-row">
                <span class="detail-label">Classification:</span>
                <span class="detail-value">{{ command.classification }}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Target:</span>
                <span class="detail-value">{{ command.target }}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">UNIX timestamp:</span>
                <span class="detail-value">{{ command.timestamp }}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Time interval:</span>
                <span class="detail-value">{{ command.timestamp_interval }}s</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Consecutive occurrence:</span>
                <span class="detail-value">{{ command.merge_count }}</span>
              </div>
            </div>
          </div>
        </transition-group>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      items: [],
      scores: [],
      socket: null,
      message: [],
      commandHistory: [], // Will store full command objects
    };
  },
  updated() {
    this.$nextTick(() => {
      const commandList = this.$refs.commandList;
      if (commandList) {
        commandList.scrollTop = commandList.scrollHeight;
      }
    });
  },
  computed: {
    rankedItems() {
      return this.items
        .map((item, index) => ({ name: item, score: this.scores[index] }))
        .sort((a, b) => b.score - a.score);
    }
  },
  methods: {
    formatTimestamp(timestamp) {
      const date = new Date(timestamp);
      return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
    }
  },
  mounted() {
    this.socket = new WebSocket('ws://localhost:8000/ws');
    
    this.socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.items = data.item;
      this.scores = data.item_scores;
      
      if (data.commandHistory) {
        this.commandHistory = data.commandHistory;
      } else {
        // Fallback if server doesn't provide full objects yet
        this.commandHistory = this.message.map((msg, index) => ({
          message: msg,
          timestamp: Date.now() - (this.message.length - index) * 1000, // Dummy timestamp
          classification: 'Unknown',
          target: 'Unknown',
          timestamp_interval: 1000,
          merge_count: 1,
          cat: 'default'
        }));
      }
    };
  },
  beforeDestroy() {
    if (this.socket) {
      this.socket.close();
    }
  },
};
</script>

<style>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.main-container {
  width: 100%;
  max-width: 1200px;
  height: 100vh;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  background-color: #f9fafc;
}

.recommendation-panel {
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  padding: 20px;
  border: 1px solid #e8eef3;
}

.title {
  color: #2d3748;
  font-size: 1.75rem;
  font-weight: 700;
  text-align: center;
  margin-bottom: 16px;
}

.recommendation-content {
  background-color: #f7f9fc;
  border-radius: 8px;
  padding: 16px;
}

.recommendation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.subtitle {
  color: #4a5568;
  font-size: 1.1rem;
  font-weight: 600;
}

.badge-count {
  background: #3182ce;
  color: white;
  height: 26px;
  min-width: 26px;
  border-radius: 13px;
  font-size: 0.85rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 8px;
}

.recommendation-list {
  max-height: 300px;
  overflow-y: auto;
}

.recommendation-item {
  display: flex;
  align-items: center;
  background: white;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 8px;
  border-left: 4px solid #cbd5e0;
  transition: all 0.3s ease;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.recommendation-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.top-item {
  border-left: 4px solid #4299e1;
  background-color: #ebf8ff;
}

.item-rank {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 28px;
  height: 28px;
  background: #edf2f7;
  border-radius: 50%;
  font-weight: 600;
  font-size: 0.9rem;
  color: #4a5568;
  margin-right: 12px;
}

.top-item .item-rank {
  background: #bee3f8;
  color: #2b6cb0;
}

.item-name {
  flex-grow: 1;
  font-size: 1rem;
  font-weight: 500;
  color: #2d3748;
}

.item-score {
  font-size: 0.85rem;
  font-weight: 600;
  color: #718096;
  background: #edf2f7;
  padding: 4px 8px;
  border-radius: 4px;
}

.top-item .item-score {
  background: #bee3f8;
  color: #2b6cb0;
}

.historical-commands-container {
  flex-grow: 1;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  border: 1px solid #e8eef3;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #e8eef3;
}

.historical-commands-title {
  color: #2d3748;
  font-size: 1.1rem;
  font-weight: 600;
}

.command-count {
  color: #718096;
  font-size: 0.9rem;
  font-weight: 500;
  background: #edf2f7;
  padding: 4px 12px;
  border-radius: 16px;
}

.historical-commands-list {
  overflow-y: auto;
  flex-grow: 1;
  scrollbar-width: thin;
  scroll-behavior: smooth;
  padding: 16px;
}

.command-card {
  background-color: #ffffff;
  border: 1px solid #edf2f7;
  border-radius: 6px;
  margin-bottom: 10px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  transition: all 0.2s ease;
  overflow: hidden;
  font-size: 0.85rem; 
}

.command-card:hover {
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
}

.command-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px; 
  background-color: #f7f9fc;
  border-bottom: 1px solid #edf2f7;
}

.command-name {
  font-weight: 600;
  font-size: 0.95rem; 
  color: #2d3748;
}

.command-cat {
  padding: 3px 6px; 
  border-radius: 10px;
  font-size: 0.7rem;
  font-weight: 600;
}

.command-details {
  padding: 6px 12px; 
}

.detail-row {
  display: flex;
  margin-bottom: 4px; 
  font-size: 0.75rem; 
  align-items: center;
}

.detail-label {
  width: 160px; 
  color: #718096;
  font-weight: 500;
  white-space: nowrap;
}

.detail-value {
  flex-grow: 1;
  color: #4a5568;
}

.cat-default { background-color: #edf2f7; color: #4a5568; }
.cat-system { background-color: #c6f6d5; color: #2f855a; }
.cat-user { background-color: #bee3f8; color: #2b6cb0; }
.cat-app { background-color: #e9d8fd; color: #6b46c1; }
.cat-file { background-color: #feebc8; color: #c05621; }
.cat-network { background-color: #b2f5ea; color: #0987a0; }


.list-enter-active, .list-leave-active {
  transition: all 0.5s ease;
  position: relative;
}
.list-enter, .list-leave-to {
  opacity: 0;
  transform: translateX(30px);
  position: sticky;
}

.message-enter-active, .message-leave-active {
  transition: all 0.4s ease;
}
.message-enter, .message-leave-to {
  opacity: 0;
  transform: translateY(20px);
}
</style>