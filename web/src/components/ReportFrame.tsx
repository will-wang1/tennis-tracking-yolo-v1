"use client";

import { useEffect, useRef, useState } from "react";

/** The full report page in a frame that grows to fit it, so the session
 * page scrolls as one page (the frame is same-origin, so its height can be
 * read; it changes as sections open and charts draw). */
export function ReportFrame({ src, title }: { src: string; title: string }) {
  const ref = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(1200);

  useEffect(() => {
    const frame = ref.current;
    if (!frame) return;
    let observer: ResizeObserver | undefined;
    const fit = () => {
      const doc = frame.contentDocument;
      if (!doc?.body) return;
      setHeight(doc.documentElement.scrollHeight);
      observer?.disconnect();
      observer = new ResizeObserver(() => setHeight(doc.documentElement.scrollHeight));
      observer.observe(doc.body);
    };
    frame.addEventListener("load", fit);
    fit();
    return () => { frame.removeEventListener("load", fit); observer?.disconnect(); };
  }, []);

  return <iframe ref={ref} src={src} title={title} className="report-frame" style={{ height }} />;
}
