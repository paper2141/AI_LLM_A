/**
 * Code.gs - Google Apps Script 백엔드 (Cloud Run RAG 서버 연동)
 *
 * 설정: 프로젝트 설정 > 스크립트 속성 에 아래 값을 추가한다. (코드에 URL 을 쓰지 않음)
 *   CLOUD_RUN_URL : 예) https://rag-server-xxxxx-an.a.run.app  (끝에 / 없이)
 *   API_SECRET    : (선택) 서버에 API_SECRET 을 설정했다면 같은 값
 *
 * 서버의 StreamHandler 는 SSE(text/event-stream)로 토큰을 흘려보낸다.
 * Apps Script 의 UrlFetchApp 은 응답을 끝까지 받은 뒤에야 반환하므로(실시간 중계 불가),
 * 여기서 SSE 를 이벤트 배열로 파싱해 클라이언트에 넘기고,
 * index.html 이 이벤트를 순서대로 재생하며 스트리밍처럼 표시한다.
 */

/** 웹앱 진입점 */
function doGet() {
  return HtmlService.createHtmlOutputFromFile('index')
    .setTitle('도시 정책 RAG')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

/** 스크립트 속성에서 서버 URL 을 읽는다. 없으면 원인을 설명하는 오류를 던진다. */
function getBaseUrl_() {
  var url = PropertiesService.getScriptProperties().getProperty('CLOUD_RUN_URL');
  if (!url) {
    throw new Error('스크립트 속성 CLOUD_RUN_URL 이 설정되지 않았습니다. ' +
      '프로젝트 설정 > 스크립트 속성에 Cloud Run 서비스 URL 을 추가하세요.');
  }
  return url.replace(/\/+$/, '');
}

/** 공통 요청 옵션 (API_SECRET 이 있으면 X-API-Key 헤더 추가) */
function buildOptions_(extra) {
  var headers = {};
  var secret = PropertiesService.getScriptProperties().getProperty('API_SECRET');
  if (secret) headers['X-API-Key'] = secret;
  var options = { muteHttpExceptions: true, headers: headers };
  for (var k in extra) options[k] = extra[k];
  return options;
}

/** 서버 호출 + 네트워크/HTTP 오류를 상세 메시지로 변환 */
function fetchServer_(path, extra) {
  var url = getBaseUrl_() + path;
  var res;
  try {
    res = UrlFetchApp.fetch(url, buildOptions_(extra));
  } catch (e) {
    throw new Error('Cloud Run 서버에 연결하지 못했습니다 (' + url + '): ' + e.message +
      ' / URL 이 올바른지, 서비스가 배포되어 있는지 확인하세요.');
  }
  return res;
}

/** 일반 JSON 응답 처리. 서버가 보낸 {error:{stage,message,hint}} 를 문장으로 풀어준다. */
function parseJson_(res) {
  var code = res.getResponseCode();
  var text = res.getContentText();
  var body;
  try { body = JSON.parse(text); } catch (e) {
    throw new Error('서버 응답(HTTP ' + code + ')이 JSON 이 아닙니다: ' + text.substring(0, 300));
  }
  if (code >= 400) throw new Error(formatError_(body.error || body.detail || body, code));
  return body;
}

function formatError_(err, code) {
  if (err && err.stage) {
    return '[' + err.stage + '] ' + err.message + (err.hint ? ' → ' + err.hint : '');
  }
  return 'HTTP ' + code + ': ' + JSON.stringify(err).substring(0, 300);
}

/** SSE 본문을 이벤트 객체 배열로 파싱 */
function parseSse_(res) {
  var code = res.getResponseCode();
  var text = res.getContentText();
  if (code >= 400) {
    var body;
    try { body = JSON.parse(text); } catch (e) { body = text.substring(0, 300); }
    // 422(Pydantic 검증 오류) 등도 읽기 쉽게 변환
    throw new Error(formatError_(body.error || body.detail || body, code));
  }
  var events = [];
  text.split('\n\n').forEach(function (block) {
    var line = block.trim();
    if (line.indexOf('data:') !== 0) return;
    try { events.push(JSON.parse(line.substring(5).trim())); } catch (e) { /* 깨진 조각은 무시 */ }
  });
  return events;
}

/** 서버 연동 상태 (index.html 이 주기적으로 호출) */
function getServerStatus() {
  try {
    var res = fetchServer_('/health', { method: 'get' });
    var body = parseJson_(res);
    body.reachable = true;
    return body;
  } catch (e) {
    return { reachable: false, ok: false, error: e.message };
  }
}

/** PDF(base64) -> 서버에서 페이지별 텍스트 추출 */
function extractPdf(base64, filename) {
  var blob = Utilities.newBlob(Utilities.base64Decode(base64), 'application/pdf', filename);
  var res = fetchServer_('/extract', { method: 'post', payload: { file: blob } });
  return parseJson_(res);
}

/** 벡터 DB 생성: 서버의 진행 이벤트(progress/done/error) 배열 반환 */
function ingestDocument(source, pages) {
  var res = fetchServer_('/ingest', {
    method: 'post', contentType: 'application/json',
    payload: JSON.stringify({ source: source, pages: pages }),
  });
  return parseSse_(res);
}

/** 질의: 서버 StreamHandler 가 만든 이벤트(augmented/sources/token/done/error) 배열 반환 */
function askQuestion(question, history, source) {
  var res = fetchServer_('/query', {
    method: 'post', contentType: 'application/json',
    payload: JSON.stringify({ question: question, history: history || [], source: source || null }),
  });
  return parseSse_(res);
}
