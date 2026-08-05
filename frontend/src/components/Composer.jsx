import { useEffect, useRef, useState } from "react";
import { CloseIcon, ImageIcon, SendIcon } from "./Icons.jsx";

export default function Composer({ value, onChange, image, onImage, onSend, disabled }) {
  const inputRef = useRef(null);
  const [preview, setPreview] = useState(null);
  useEffect(() => {
    if (!image) {
      setPreview(null);
      return undefined;
    }
    const url = URL.createObjectURL(image);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);
  const submit = (event) => {
    event.preventDefault();
    if ((!value.trim() && !image) || disabled) return;
    onSend();
  };

  return (
    <form className="composer" onSubmit={submit}>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        hidden
        onChange={(event) => onImage(event.target.files?.[0] || null)}
      />
      <button className="attach-button" type="button" onClick={() => inputRef.current?.click()} aria-label="上传叶片照片" disabled={disabled}>
        <ImageIcon />
      </button>
      {preview ? (
        <div className="composer-preview">
          <img src={preview} alt="待上传图片预览" />
          <button type="button" onClick={() => onImage(null)} aria-label="移除图片"><CloseIcon /></button>
        </div>
      ) : null}
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            submit(event);
          }
        }}
        rows="1"
        maxLength="4000"
        placeholder="描述症状，或上传叶片照片…"
        disabled={disabled}
        aria-label="问题"
      />
      <span className="character-count">{value.length}/4000</span>
      <button className="send-button" type="submit" disabled={disabled || (!value.trim() && !image)}>
        <SendIcon />
        {disabled ? "生成中" : "发送"}
      </button>
    </form>
  );
}
