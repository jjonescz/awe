import { existsSync } from 'fs';
import { PageRecipe } from './page-recipe';

export enum ValidationResultType {
  Valid,
  /** Extracted HTML doesn't exist. */
  HtmlNonExistent,
  /** Extracted JSON doesn't exist. */
  JsonNonExistent,
}

export interface MissingGroundTruth {
  /** Ground-truth label not found in the extracted HTML. */
  readonly label: string;
  /** Ground-truth field value not found in the extracted HTML. */
  readonly value: string;
}

export class ValidationResult {
  public constructor(
    public readonly type: ValidationResultType,
    public readonly data: MissingGroundTruth | null = null
  ) {}

  public toString() {
    let s = ValidationResultType[this.type];
    if (this.data !== null) {
      s += `(${Object.entries(this.data)
        .map(([k, v]) => `${k}=${v}`)
        .join(', ')})`;
    }
    return s;
  }
}

/** Validates that extracted pages contain expected ground-truth data. */
export class Validator {
  public constructor(public readonly recipe: PageRecipe) {}

  public async validate(): Promise<ValidationResult> {
    if (!existsSync(this.recipe.htmlPath)) {
      return new ValidationResult(ValidationResultType.HtmlNonExistent);
    }

    if (!existsSync(this.recipe.jsonPath)) {
      return new ValidationResult(ValidationResultType.JsonNonExistent);
    }

    return new ValidationResult(ValidationResultType.Valid);
  }
}
