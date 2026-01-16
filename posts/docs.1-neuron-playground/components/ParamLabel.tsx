import InfoPopover from "./InfoPopover";

type ParamLabelProps = {
  label: string;
  value: string;
  info: string;
};

export default function ParamLabel({ label, value, info }: ParamLabelProps) {
  return (
    <label className="flex items-center justify-between text-xs">
      <span className="flex items-center">
        {label}
        <InfoPopover text={info} />
      </span>
      <span>{value}</span>
    </label>
  );
}
