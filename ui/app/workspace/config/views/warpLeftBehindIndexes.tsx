import { Alert, AlertDescription } from "@/components/ui/alert";
import {
	AlertDialog,
	AlertDialogAction,
	AlertDialogCancel,
	AlertDialogContent,
	AlertDialogDescription,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogTitle,
} from "@/components/ui/alertDialog";
import { Button } from "@/components/ui/button";
import { getErrorMessage } from "@/lib/store";
import { useDeleteWarpLeftBehindIndexMutation, useGetWarpLeftBehindIndexQuery } from "@/lib/store/apis/warpApi";
import { CircleAlert, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";

interface Props {
	/** Warp uses a store of its own and the viewer can delete: only then is there anything to ask. */
	enabled: boolean;
}

/**
 * The log index Warp left in the shared vector store when warp.vector_store
 * gave it a store of its own. The server works it out from the stores on each
 * read, and deleting it refetches only this, so a draft on the settings form
 * is untouched. A check that failed says so, with a retry: it is not the same
 * as finding nothing. It renders inside the settings form, so its buttons are
 * type="button": a cleanup must not also save the settings.
 */
export default function WarpLeftBehindIndexes({ enabled }: Props) {
	const { t } = useTranslation("config");
	const { t: tc } = useTranslation("common");
	const [confirming, setConfirming] = useState(false);
	const { data, error, isFetching, refetch } = useGetWarpLeftBehindIndexQuery(undefined, { skip: !enabled });
	const [deleteIndex, { isLoading: isDeleting }] = useDeleteWarpLeftBehindIndexMutation();
	const leftover = enabled ? data?.left_behind : null;

	if (enabled && error && !leftover) {
		return (
			<div className="mb-4">
				<Alert variant="destructive" data-testid="warp-left-behind-index-error">
					<CircleAlert className="h-4 w-4" />
					<AlertDescription className="gap-2">
						<span>{t("warp.leftBehind.checkFailed", { error: getErrorMessage(error) })}</span>
						<div>
							<Button
								type="button"
								variant="outline"
								size="sm"
								data-testid="warp-left-behind-index-retry-button"
								disabled={isFetching}
								onClick={() => void refetch()}
							>
								{tc("retry")}
							</Button>
						</div>
					</AlertDescription>
				</Alert>
			</div>
		);
	}
	if (!leftover) {
		return null;
	}

	const handleDelete = async () => {
		try {
			await deleteIndex().unwrap();
			toast.success(t("warp.leftBehind.deleted"));
		} catch (error) {
			toast.error(getErrorMessage(error));
		}
	};

	return (
		<div className="mb-4">
			<Alert variant="warning" data-testid="warp-left-behind-index-shared">
				<TriangleAlert className="h-4 w-4" />
				<AlertDescription className="gap-2">
					<span>{t("warp.leftBehind.shared", { namespace: leftover.namespace })}</span>
					<div>
						<Button
							type="button"
							variant="outline"
							size="sm"
							data-testid="warp-left-behind-index-delete-button"
							disabled={isDeleting}
							onClick={() => setConfirming(true)}
						>
							{t("warp.leftBehind.delete")}
						</Button>
					</div>
				</AlertDescription>
			</Alert>
			<AlertDialog open={confirming} onOpenChange={setConfirming}>
				<AlertDialogContent>
					<AlertDialogHeader>
						<AlertDialogTitle>{t("warp.leftBehind.confirmTitle")}</AlertDialogTitle>
						<AlertDialogDescription>{t("warp.leftBehind.confirmDescription", { namespace: leftover.namespace })}</AlertDialogDescription>
					</AlertDialogHeader>
					<AlertDialogFooter>
						<AlertDialogCancel data-testid="warp-left-behind-index-cancel-button">{tc("cancel")}</AlertDialogCancel>
						<AlertDialogAction
							data-testid="warp-left-behind-index-confirm-button"
							onClick={() => {
								setConfirming(false);
								void handleDelete();
							}}
						>
							{t("warp.leftBehind.delete")}
						</AlertDialogAction>
					</AlertDialogFooter>
				</AlertDialogContent>
			</AlertDialog>
		</div>
	);
}