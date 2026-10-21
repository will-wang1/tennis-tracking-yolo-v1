import { requireUser, isHead } from "@/lib/auth";
import { listCoaches } from "@/lib/db";
import { UploadForm } from "@/components/UploadForm";

export const dynamic = "force-dynamic";

export default async function UploadPage() {
  const user = await requireUser();
  const head = isHead(user);
  return (
    <div style={{ maxWidth: 720 }}>
      <p className="eyebrow">New session</p>
      <h1>Upload a session</h1>
      <p className="lede">The video from the court camera. It is analysed automatically and appears under {head ? "the coach's" : "your"} sessions when done - about 8 minutes of processing for each minute of video.</p>
      {!head && user.coach_id == null
        ? <p className="banner info">Your account is not linked to a coach yet - ask your head coach.</p>
        : <UploadForm coaches={head ? listCoaches().filter((c) => !c.is_demo).map((c) => ({ id: c.id, name: c.name })) : null}
            fixedCoach={user.coach_name} />}
    </div>
  );
}
