import Link from "next/link";
import { requireUser, isHead } from "@/lib/auth";
import { listUploads, TOTAL_STEPS } from "@/lib/uploads";
import { UploadList } from "@/components/UploadList";

export const dynamic = "force-dynamic";

export default async function UploadsPage() {
  const user = await requireUser();
  const head = isHead(user);
  const uploads = head ? listUploads() : user.coach_id ? listUploads(user.coach_id) : [];
  return (
    <>
      <div className="head">
        <div>
          <p className="eyebrow">{head ? "All coaches" : user.coach_name}</p>
          <h1>Uploads</h1>
          <p className="lede">Sessions on their way to a report. This page updates by itself.</p>
        </div>
        <Link href="/upload" className="btn primary">Upload a session</Link>
      </div>
      <UploadList initial={uploads} totalSteps={TOTAL_STEPS} showCoach={head} />
    </>
  );
}
