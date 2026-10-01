// Install through browser CDP using /config from the local receiver.
// Save pending, then acknowledge it before requesting another node.
(() => {
  if (window.__gtoCaseCapture) throw new Error('Capture already installed');
  const config = window.__gtoCaptureConfig;
  if (!config?.caseId || config.blockedReason) throw new Error(config?.blockedReason || 'Missing case config');
  const state = window.__gtoCaseCapture = {
    requestCount: config.requestsUsed, requestReady: false, running: false, pending: null, stopReason: null, sourceUsage: null,
  };
  let template;
  const requests = new WeakMap();
  const originalOpen = XMLHttpRequest.prototype.open;
  const originalSetHeader = XMLHttpRequest.prototype.setRequestHeader;
  const isSolutionRequest = url => url.origin === 'https://api.gtowizard.com' &&
    url.pathname === '/v4/solutions/spot-solution/';
  const matchesCase = url => isSolutionRequest(url) &&
    url.searchParams.get('gametype') === config.gameType &&
    url.searchParams.get('depth') === String(config.depth) &&
    ['board', 'stacks', 'flop_actions', 'turn_actions', 'river_actions'].every(key => !url.searchParams.get(key));
  // Use a fresh dedicated source tab so these bootstrap requests are counted once.
  state.requestCount += performance.getEntriesByType('resource')
    .filter(entry => isSolutionRequest(new URL(entry.name))).length;
  const parse = text => {
    try { return {response: JSON.parse(text)}; }
    catch {
      const clean = text.replace(/"(?:\\.|[^"\\])*"|\bNaN\b|-?Infinity\b/g,
        token => token.startsWith('"') ? token : 'null');
      return {response: JSON.parse(clean), rawText: text};
    }
  };
  const guard = () => {
    if (state.sourceUsage?.limit > 0 && state.sourceUsage.count >= state.sourceUsage.limit)
      state.stopReason = 'Source daily browsing limit reached';
    if (/daily solution browsing limit|security protection|security reasons?|verify you are human|captcha/i.test(document.body.innerText))
      state.stopReason = 'Source limit or security prompt';
    if (state.stopReason) throw new Error(state.stopReason);
  };
  // Authentication headers stay in this page closure and are never exported.
  XMLHttpRequest.prototype.open = function(method, address, ...rest) {
    const url = new URL(address, location.href);
    if (isSolutionRequest(url)) state.requestCount++;
    if (matchesCase(url)) {
      const request = {method, url, headers: {}};
      requests.set(this, request);
      this.addEventListener('load', () => {
        if (this.status !== 200) { state.stopReason = 'Source HTTP ' + this.status; return; }
        try {
          const response = this.responseType === 'json' ? this.response : parse(this.responseText).response;
          state.sourceUsage = response.usage || state.sourceUsage;
        } catch (error) { state.stopReason = String(error); return; }
        request.credentials = this.withCredentials ? 'include' : 'omit';
        template = request;
        state.requestReady = true;
      });
      for (const event of ['error', 'timeout', 'abort']) this.addEventListener(event, () => {
        state.stopReason = 'Source request ' + event;
      });
    }
    return originalOpen.call(this, method, address, ...rest);
  };
  XMLHttpRequest.prototype.setRequestHeader = function(name, value) {
    const request = requests.get(this);
    if (request) request.headers[name] = value;
    return originalSetHeader.call(this, name, value);
  };
  state.capture = async job => {
    if (state.running || state.pending) throw new Error('Save and acknowledge the previous node first');
    guard();
    if (!template) throw new Error('Observe a successful case request in the UI first');
    state.running = true;
    try {
      const url = new URL(template.url);
      url.searchParams.set('preflop_actions', job.path);
      if (!matchesCase(url)) throw new Error('Request outside case');
      state.requestCount++;
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 30000);
      let captured;
      try {
        const result = await fetch(url, {method: template.method, headers: template.headers,
          credentials: template.credentials, signal: controller.signal});
        if (!result.ok) throw new Error('Source HTTP ' + result.status);
        captured = parse(await result.text());
      } finally { clearTimeout(timer); }
      const response = captured.response;
      state.sourceUsage = response.usage || state.sourceUsage;
      if (response.game.current_street.type !== 'PREFLOP' || response.game.active_position !== job.expectedActor)
        throw new Error('Source street or actor mismatch');
      if (response.hands_locked || (response.warning && response.warning !== 'ZERO_RANGE'))
        throw new Error('Source node unavailable: ' + String(response.warning));
      const actor = response.game.active_position;
      if (response.warning === 'ZERO_RANGE') {
        captured = {response: {warning:'ZERO_RANGE', hands_locked:null,
          game:{active_position:actor,current_street:{type:'PREFLOP'}}}};
      } else if (!response.action_solutions.length || response.action_solutions.some(entry =>
        entry.strategy.length !== 169 || entry.evs.length !== 169)) throw new Error('Incomplete source strategy');
      const sourceUrl = new URL('/solutions', location.origin);
      for (const [key,value] of Object.entries({solution_type:'gwiz',soltab:'strategy',
        gametype:config.gameType,depth:String(config.depth),preflop_actions:job.path,
        history_spot:String(job.history.length),stratab:'strategy_ev'})) sourceUrl.searchParams.set(key,value);
      state.pending = {path:job.path,history:job.history,sourceUrl:sourceUrl.href,
        requestUrl:url.href,capturedAt:new Date().toISOString(),...captured};
      return {path:job.path,actor,zeroRange:response.warning === 'ZERO_RANGE',requestCount:state.requestCount};
    } catch (error) { state.stopReason = String(error); throw error; }
    finally { state.running = false; }
  };
  state.acknowledge = path => {
    if (state.pending?.path !== path) throw new Error('Acknowledgement mismatch');
    state.pending = null;
  };
  state.dispose = () => {
    if (state.running || state.pending) throw new Error('Stop and save before cleanup');
    XMLHttpRequest.prototype.open = originalOpen;
    XMLHttpRequest.prototype.setRequestHeader = originalSetHeader;
    template = undefined;
    delete window.__gtoCaseCapture;
    delete window.__gtoCaptureConfig;
  };
  return {installed:true,caseId:config.caseId};
})()
