export default function MessageContent({ text }) {
  const lines = text.split("\n");
  const nodes = [];
  let list = [];

  const flushList = (key) => {
    if (!list.length) return;
    nodes.push(
      <ul key={`list-${key}`}>
        {list.map((item, index) => <li key={`${key}-${index}`}>{item}</li>)}
      </ul>,
    );
    list = [];
  };

  lines.forEach((raw, index) => {
    const line = raw.trim();
    if (line.startsWith("- ") || line.startsWith("• ")) {
      list.push(line.slice(2));
      return;
    }
    flushList(index);
    if (!line) return;
    if (line.startsWith("### ")) {
      nodes.push(<h3 key={index}>{line.slice(4)}</h3>);
    } else if (line.startsWith("## ")) {
      nodes.push(<h2 key={index}>{line.slice(3)}</h2>);
    } else {
      nodes.push(<p key={index}>{line.replace(/\*\*/g, "")}</p>);
    }
  });
  flushList("end");
  return <div className="message-content">{nodes}</div>;
}
